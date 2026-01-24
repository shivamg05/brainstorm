"""
Causal preprocessing for ECoG signals.

Implements spectral feature extraction using causal (real-time compatible) filters.
Key features:
- Bandpass filtering for frequency bands (especially high-gamma 70-150Hz)
- Causal IIR filters (only use past data)
- Band power estimation
"""

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Optional


class CausalBandpassFilter:
    """
    Causal bandpass filter using IIR (Butterworth) design.

    Maintains filter state for streaming/real-time processing.
    """

    def __init__(
        self,
        low_freq: float,
        high_freq: float,
        fs: float = 1000.0,
        order: int = 4,
    ):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = fs
        self.order = order

        nyq = fs / 2
        low = low_freq / nyq
        high = high_freq / nyq

        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))

        self.b, self.a = signal.butter(order, [low, high], btype="band")

        self.zi: Optional[np.ndarray] = None
        self.n_channels: Optional[int] = None

    def reset(self, n_channels: int) -> None:
        """Reset filter state for new sequence."""
        self.n_channels = n_channels
        zi_single = signal.lfilter_zi(self.b, self.a)
        # CHANGE: make zi float64 for numerical stability in streaming filters
        self.zi = np.zeros((n_channels, len(zi_single)), dtype=np.float64)

    def filter_sample(self, x: np.ndarray) -> np.ndarray:
        """
        Filter a single sample (streaming mode).

        Args:
            x: Input sample of shape (n_channels,)

        Returns:
            Filtered sample of shape (n_channels,)
        """
        if self.zi is None or self.n_channels != len(x):
            self.reset(len(x))

        # CHANGE: compute in float64 then cast back (less drift in long streams)
        y = np.zeros_like(x, dtype=np.float64)
        for i in range(len(x)):
            y_out, self.zi[i] = signal.lfilter(self.b, self.a, [x[i]], zi=self.zi[i])
            y[i] = y_out[0]
        return y.astype(np.float32)

    def filter_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Filter a batch of samples (training mode).

        Args:
            x: Input of shape (n_samples, n_channels)

        Returns:
            Filtered signal of shape (n_samples, n_channels)
        """
        # CHANGE: explicitly cast output to float32 (keeps downstream consistent)
        return signal.lfilter(self.b, self.a, x, axis=0).astype(np.float32)


class SpectralFeatureExtractor:
    """
    Extracts spectral power features from ECoG signals.

    Uses causal bandpass filtering and power estimation for real-time processing.
    """

    DEFAULT_BANDS = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 70),
        "high_gamma": (70, 150),
    }

    def __init__(
        self,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        fs: float = 1000.0,
        smoothing_samples: int = 50,
    ):
        self.bands = bands or self.DEFAULT_BANDS
        self.fs = fs
        self.smoothing_samples = smoothing_samples

        self.filters: Dict[str, CausalBandpassFilter] = {}
        for name, (low, high) in self.bands.items():
            self.filters[name] = CausalBandpassFilter(low, high, fs)

        self.power_state: Dict[str, Optional[np.ndarray]] = {name: None for name in self.bands}

        # EMA factor (for streaming mode)
        self.alpha = 2.0 / (smoothing_samples + 1)

    def reset(self) -> None:
        """Reset all filter and smoothing states."""
        for filt in self.filters.values():
            filt.zi = None
        self.power_state = {name: None for name in self.bands}

    def extract_sample(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spectral features for a single sample (streaming mode).

        Args:
            x: Input sample of shape (n_channels,)

        Returns:
            Dictionary of {band_name: power} where power is shape (n_channels,)
        """
        features: Dict[str, np.ndarray] = {}

        for name, filt in self.filters.items():
            filtered = filt.filter_sample(x)
            power = filtered ** 2

            if self.power_state[name] is None:
                self.power_state[name] = power
            else:
                self.power_state[name] = self.alpha * power + (1 - self.alpha) * self.power_state[name]

            features[name] = self.power_state[name].astype(np.float32)

        return features

    def extract_batch(self, x: np.ndarray, return_raw: bool = False) -> np.ndarray:
        """
        Extract spectral features for a batch (training mode).

        Args:
            x: Input of shape (n_samples, n_channels)
            return_raw: If True, also return raw signal

        Returns:
            Features of shape (n_samples, n_channels * n_bands) or
            (n_samples, n_channels * (n_bands + 1)) if return_raw
        """
        n_samples, n_channels = x.shape
        n_bands = len(self.bands)

        if return_raw:
            features = np.zeros((n_samples, n_channels * (n_bands + 1)), dtype=np.float32)
            features[:, :n_channels] = x.astype(np.float32)
            offset = n_channels
        else:
            features = np.zeros((n_samples, n_channels * n_bands), dtype=np.float32)
            offset = 0

        # CHANGE: build kernel once outside the channel loop
        kernel = np.ones(self.smoothing_samples, dtype=np.float32) / self.smoothing_samples

        for i, (_name, filt) in enumerate(self.filters.items()):
            filtered = filt.filter_batch(x)
            power = filtered ** 2

            smoothed = np.zeros_like(power, dtype=np.float32)
            for ch in range(n_channels):
                smoothed[:, ch] = np.convolve(power[:, ch], kernel, mode="full")[:n_samples]

            smoothed = np.log1p(smoothed).astype(np.float32)

            start = offset + i * n_channels
            end = offset + (i + 1) * n_channels
            features[:, start:end] = smoothed

        return features


class ECoGPreprocessor:
    """
    Complete preprocessing pipeline for ECoG signals.

    Combines:
    - Z-score normalization
    - Optional PCA channel reduction (e.g., 1024 -> 64)
    - Spectral feature extraction
    """

    def __init__(
        self,
        use_spectral: bool = True,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        include_raw: bool = True,
        fs: float = 1000.0,
        n_pca_components: int = 64,
    ):
        self.use_spectral = use_spectral
        self.include_raw = include_raw
        self.fs = fs

        # CHANGE: configurable PCA component count (default 64)
        self.n_pca_components = n_pca_components

        # Normalization stats (RAW channel space)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        # CHANGE: PCA + PCA-output normalization stats
        self.pca: Optional[PCA] = None
        self.pca_mean: Optional[np.ndarray] = None
        self.pca_std: Optional[np.ndarray] = None

        # Spectral extractor operates on *post-PCA* channels if PCA is enabled
        self.spectral = SpectralFeatureExtractor(bands=bands, fs=fs) if use_spectral else None

        # CHANGE: track both raw channels and output channels (post-PCA)
        self._n_raw_channels: Optional[int] = None
        self._n_out_channels: Optional[int] = None
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray) -> "ECoGPreprocessor":
        """
        Fit normalization statistics and PCA on training data.

        Args:
            X: Training data of shape (n_samples, n_channels)
        """
        # CHANGE: sanity check (helps catch shape bugs early)
        if X.ndim != 2:
            raise ValueError(f"Expected X shape (n_samples, n_channels); got {X.shape}")

        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)
        self._n_raw_channels = int(X.shape[1])

        # CHANGE: normalize first in raw-channel space (important before PCA)
        X_norm = (X - self.mean) / (self.std + 1e-8)

        # CHANGE: fit PCA only if it actually reduces dimensionality
        if self._n_raw_channels > self.n_pca_components:
            self.pca = PCA(n_components=self.n_pca_components, random_state=42)
            X_pca = self.pca.fit_transform(X_norm)

            # CHANGE: normalize PCA outputs for stable training (optional but helpful)
            self.pca_mean = X_pca.mean(axis=0, keepdims=True).astype(np.float32)
            self.pca_std = (X_pca.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

            self._n_out_channels = int(self.n_pca_components)
        else:
            self.pca = None
            self.pca_mean = None
            self.pca_std = None
            self._n_out_channels = int(self._n_raw_channels)

        # CHANGE: compute feature dimensionality using output channel count (post-PCA)
        n_bands = len(self.spectral.bands) if self.spectral else 0
        if self.include_raw:
            self._n_features = self._n_out_channels * (1 + n_bands)
        else:
            self._n_features = self._n_out_channels * n_bands

        return self

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            raise RuntimeError("Call fit() first")
        return self._n_features

    # CHANGE: expose post-PCA channel count (useful for model wiring)
    @property
    def n_out_channels(self) -> int:
        if self._n_out_channels is None:
            raise RuntimeError("Call fit() first")
        return self._n_out_channels

    # CHANGE: helper to apply (normalize -> PCA -> PCA-normalize) for batch
    def _apply_norm_and_pca_batch(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() first")

        X_norm = (X - self.mean) / (self.std + 1e-8)

        if self.pca is not None:
            X_pca = self.pca.transform(X_norm).astype(np.float32)
            X_pca = (X_pca - self.pca_mean) / self.pca_std  # type: ignore[operator]
            return X_pca.astype(np.float32)

        return X_norm.astype(np.float32)

    # CHANGE: helper to apply (normalize -> PCA -> PCA-normalize) for one sample
    def _apply_norm_and_pca_sample(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() first")

        x_norm = (x - self.mean) / (self.std + 1e-8)

        if self.pca is not None:
            x_pca = self.pca.transform(x_norm.reshape(1, -1))[0].astype(np.float32)
            x_pca = (x_pca - self.pca_mean.reshape(-1)) / self.pca_std.reshape(-1)  # type: ignore[union-attr]
            return x_pca.astype(np.float32)

        return x_norm.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data (batch mode for training).

        Args:
            X: Input of shape (n_samples, n_channels)

        Returns:
            Features of shape (n_samples, n_features)
        """
        # CHANGE: apply PCA pipeline before spectral features
        X_proc = self._apply_norm_and_pca_batch(X)

        if self.spectral is None:
            return X_proc

        return self.spectral.extract_batch(X_proc, return_raw=self.include_raw).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def reset_streaming(self) -> None:
        """Reset state for streaming inference."""
        if self.spectral:
            self.spectral.reset()

    def transform_sample(self, x: np.ndarray) -> np.ndarray:
        """
        Transform a single sample (streaming mode).

        Args:
            x: Input of shape (n_channels,)

        Returns:
            Features of shape (n_features,)
        """
        # CHANGE: apply PCA pipeline before spectral features
        x_proc = self._apply_norm_and_pca_sample(x)

        if self.spectral is None:
            return x_proc

        band_features = self.spectral.extract_sample(x_proc)

        # CHANGE: concatenate raw (post-PCA) + log1p(power) in band order
        parts = [x_proc] if self.include_raw else []
        for name in self.spectral.bands:
            parts.append(np.log1p(band_features[name]).astype(np.float32))

        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------
    # CHANGE: Save/Load utilities so your model checkpoint can restore PCA
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """
        Return a fully-serializable state (enough to reproduce transform()).
        """
        state = {
            "use_spectral": self.use_spectral,
            "include_raw": self.include_raw,
            "fs": self.fs,
            "n_pca_components": self.n_pca_components,
            "mean": self.mean,
            "std": self.std,
            "_n_raw_channels": self._n_raw_channels,
            "_n_out_channels": self._n_out_channels,
            "_n_features": self._n_features,
            "pca_mean": self.pca_mean,
            "pca_std": self.pca_std,
        }

        if self.pca is None:
            state["pca"] = None
        else:
            # CHANGE: store PCA params needed for transform() (components_ + mean_)
            state["pca"] = {
                "components_": self.pca.components_.astype(np.float32),
                "mean_": self.pca.mean_.astype(np.float32),
                "explained_variance_": getattr(self.pca, "explained_variance_", None),
                "n_features_in_": getattr(self.pca, "n_features_in_", None),
            }
            if state["pca"]["explained_variance_"] is not None:
                state["pca"]["explained_variance_"] = state["pca"]["explained_variance_"].astype(np.float32)

        return state

    def set_state(self, state: dict) -> None:
        """
        Restore from get_state().
        """
        self.use_spectral = bool(state["use_spectral"])
        self.include_raw = bool(state["include_raw"])
        self.fs = float(state["fs"])
        self.n_pca_components = int(state["n_pca_components"])

        self.mean = state["mean"]
        self.std = state["std"]
        self._n_raw_channels = state["_n_raw_channels"]
        self._n_out_channels = state["_n_out_channels"]
        self._n_features = state["_n_features"]

        self.pca_mean = state["pca_mean"]
        self.pca_std = state["pca_std"]

        pca_state = state.get("pca", None)
        if pca_state is None:
            self.pca = None
        else:
            # CHANGE: rebuild PCA object with required attributes for transform()
            self.pca = PCA(n_components=self.n_pca_components, random_state=42)
            self.pca.components_ = pca_state["components_"]
            self.pca.mean_ = pca_state["mean_"]
            if pca_state.get("explained_variance_") is not None:
                self.pca.explained_variance_ = pca_state["explained_variance_"]
            if pca_state.get("n_features_in_") is not None:
                self.pca.n_features_in_ = int(pca_state["n_features_in_"])

        # CHANGE: ensure spectral extractor exists iff use_spectral True
        if self.use_spectral and self.spectral is None:
            self.spectral = SpectralFeatureExtractor(fs=self.fs)
        if (not self.use_spectral) and self.spectral is not None:
            self.spectral = None
