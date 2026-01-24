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
from typing import Dict, List, Tuple, Optional


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
        """
        Initialize the bandpass filter.

        Args:
            low_freq: Lower cutoff frequency (Hz)
            high_freq: Upper cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order (higher = sharper cutoff but more delay)
        """
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = fs
        self.order = order

        # Design Butterworth bandpass filter
        nyq = fs / 2
        low = low_freq / nyq
        high = high_freq / nyq

        # Clip to valid range
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))

        self.b, self.a = signal.butter(order, [low, high], btype='band')

        # Filter state for streaming (per channel)
        self.zi: Optional[np.ndarray] = None
        self.n_channels: Optional[int] = None

    def reset(self, n_channels: int) -> None:
        """Reset filter state for new sequence."""
        self.n_channels = n_channels
        # Initialize filter state to steady-state for zero input
        zi_single = signal.lfilter_zi(self.b, self.a)
        self.zi = np.zeros((n_channels, len(zi_single)))

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

        # Filter each channel
        y = np.zeros_like(x)
        for i in range(len(x)):
            y_out, self.zi[i] = signal.lfilter(
                self.b, self.a, [x[i]], zi=self.zi[i]
            )
            y[i] = y_out[0]  # Extract scalar from 1-element array
        return y

    def filter_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Filter a batch of samples (training mode).

        Args:
            x: Input of shape (n_samples, n_channels)

        Returns:
            Filtered signal of shape (n_samples, n_channels)
        """
        # Use causal filtering (lfilter, not filtfilt)
        return signal.lfilter(self.b, self.a, x, axis=0)


class SpectralFeatureExtractor:
    """
    Extracts spectral power features from ECoG signals.

    Uses causal bandpass filtering and power estimation for real-time processing.
    """

    # Standard ECoG frequency bands
    DEFAULT_BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 70),
        'high_gamma': (70, 150),
    }

    def __init__(
        self,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        fs: float = 1000.0,
        smoothing_samples: int = 50,
    ):
        """
        Initialize the spectral feature extractor.

        Args:
            bands: Dictionary of {band_name: (low_freq, high_freq)}
            fs: Sampling frequency (Hz)
            smoothing_samples: Number of samples for power smoothing
        """
        self.bands = bands or self.DEFAULT_BANDS
        self.fs = fs
        self.smoothing_samples = smoothing_samples

        # Create filters for each band
        self.filters: Dict[str, CausalBandpassFilter] = {}
        for name, (low, high) in self.bands.items():
            self.filters[name] = CausalBandpassFilter(low, high, fs)

        # Power smoothing state (exponential moving average)
        self.power_state: Dict[str, Optional[np.ndarray]] = {
            name: None for name in self.bands
        }

        # Smoothing factor (alpha for EMA)
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
        features = {}

        for name, filt in self.filters.items():
            # Bandpass filter
            filtered = filt.filter_sample(x)

            # Instantaneous power (squared amplitude)
            power = filtered ** 2

            # Smooth with exponential moving average
            if self.power_state[name] is None:
                self.power_state[name] = power
            else:
                self.power_state[name] = (
                    self.alpha * power +
                    (1 - self.alpha) * self.power_state[name]
                )

            features[name] = self.power_state[name]

        return features

    def extract_batch(
        self,
        x: np.ndarray,
        return_raw: bool = False
    ) -> np.ndarray:
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

        # Allocate output
        if return_raw:
            features = np.zeros((n_samples, n_channels * (n_bands + 1)))
            features[:, :n_channels] = x  # Raw signal
            offset = n_channels
        else:
            features = np.zeros((n_samples, n_channels * n_bands))
            offset = 0

        for i, (name, filt) in enumerate(self.filters.items()):
            # Bandpass filter (causal)
            filtered = filt.filter_batch(x)

            # Power (squared amplitude)
            power = filtered ** 2

            # Smooth power with causal moving average
            # Use cumsum trick for efficiency
            kernel = np.ones(self.smoothing_samples) / self.smoothing_samples
            smoothed = np.zeros_like(power)
            for ch in range(n_channels):
                # Causal convolution (only past samples)
                smoothed[:, ch] = np.convolve(
                    power[:, ch], kernel, mode='full'
                )[:n_samples]

            # Log power (more Gaussian distribution)
            smoothed = np.log1p(smoothed)

            features[:, offset + i*n_channels : offset + (i+1)*n_channels] = smoothed

        return features


class ECoGPreprocessor:
    """
    Complete preprocessing pipeline for ECoG signals.

    Combines:
    - Z-score normalization
    - Spectral feature extraction
    - Optional channel selection
    """

    def __init__(
        self,
        use_spectral: bool = True,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        include_raw: bool = True,
        fs: float = 1000.0,
    ):
        """
        Initialize the preprocessor.

        Args:
            use_spectral: Whether to extract spectral features
            bands: Frequency bands to extract (None = defaults)
            include_raw: Whether to include raw (normalized) signal
            fs: Sampling frequency
        """
        self.use_spectral = use_spectral
        self.include_raw = include_raw
        self.fs = fs

        # Normalization stats
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        # Spectral extractor
        if use_spectral:
            self.spectral = SpectralFeatureExtractor(bands=bands, fs=fs)
        else:
            self.spectral = None

        # Feature info
        self._n_raw_channels: Optional[int] = None
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray) -> 'ECoGPreprocessor':
        """
        Fit normalization statistics on training data.

        Args:
            X: Training data of shape (n_samples, n_channels)

        Returns:
            self
        """
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self._n_raw_channels = X.shape[1]

        # Calculate output feature size
        n_bands = len(self.spectral.bands) if self.spectral else 0
        if self.include_raw:
            self._n_features = self._n_raw_channels * (1 + n_bands)
        else:
            self._n_features = self._n_raw_channels * n_bands

        return self

    @property
    def n_features(self) -> int:
        """Number of output features per sample."""
        if self._n_features is None:
            raise RuntimeError("Call fit() first")
        return self._n_features

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data (batch mode for training).

        Args:
            X: Input of shape (n_samples, n_channels)

        Returns:
            Features of shape (n_samples, n_features)
        """
        if self.mean is None:
            raise RuntimeError("Call fit() first")

        # Normalize
        X_norm = (X - self.mean) / (self.std + 1e-8)

        if self.spectral is None:
            return X_norm

        # Extract spectral features
        return self.spectral.extract_batch(X_norm, return_raw=self.include_raw)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
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
        if self.mean is None:
            raise RuntimeError("Call fit() first")

        # Normalize
        x_norm = (x - self.mean) / (self.std + 1e-8)

        if self.spectral is None:
            return x_norm

        # Extract spectral features
        band_features = self.spectral.extract_sample(x_norm)

        # Concatenate
        if self.include_raw:
            features = [x_norm]
        else:
            features = []

        for name in self.spectral.bands:
            # Log power
            features.append(np.log1p(band_features[name]))

        return np.concatenate(features)