from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from brainstorm.constants import GRID_HEIGHT, GRID_WIDTH, N_CHANNELS


@dataclass(frozen=True)
class NormalizerStats:
    mean: np.ndarray
    std: np.ndarray


def common_average_reference(x: np.ndarray) -> np.ndarray:
    """Apply CAR per time step: input (T, C), output (T, C)."""
    x = x.astype(np.float32, copy=False)
    return x - x.mean(axis=1, keepdims=True)


def design_bandpass_sos(
    bands_hz: Sequence[tuple[float, float]],
    fs_hz: float,
    order: int = 4,
) -> list[np.ndarray]:
    """Design SOS filters: bands_hz (B,2) -> list of B SOS arrays."""
    nyq = fs_hz / 2.0
    sos_bank: list[np.ndarray] = []
    for lo_hz, hi_hz in bands_hz:
        lo = lo_hz / nyq
        hi = hi_hz / nyq
        sos = butter(N=order, Wn=[lo, hi], btype="bandpass", output="sos")
        sos_bank.append(sos)
    return sos_bank


def compute_log_power_features(
    x: np.ndarray,
    sos_bank: Iterable[np.ndarray],
    fs_hz: float,
    tau_s: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Bandpass -> square -> EMA -> log.
    Inputs: x (T, C), sos_bank (B), fs_hz, tau_s.
    Output: features (T, B, C).
    """
    x = x.astype(np.float32, copy=False)
    n_samples, n_ch = x.shape
    sos_list = list(sos_bank)
    n_bands = len(sos_list)
    power_feat = np.empty((n_samples, n_bands, n_ch), dtype=np.float32)

    alpha = 1.0 - np.exp(-1.0 / (fs_hz * tau_s))
    b = np.array([alpha], dtype=np.float32)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float32)

    for b_idx, sos in enumerate(sos_list):
        y = sosfilt(sos, x, axis=0).astype(np.float32)
        p = y * y
        env = lfilter(b, a, p, axis=0).astype(np.float32)
        power_feat[:, b_idx, :] = np.log(env + eps)

    return power_feat


def fit_normalizer(features: np.ndarray, eps: float = 1e-6) -> NormalizerStats:
    """Compute stats: input features (T, B, C), output mean/std (1, B, C)."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + eps
    return NormalizerStats(mean=mean, std=std)


def apply_normalizer(features: np.ndarray, stats: NormalizerStats) -> np.ndarray:
    """Apply stats: input features (T, B, C), output normalized (T, B, C)."""
    return (features - stats.mean) / stats.std


def preprocess_features(
    x: np.ndarray,
    sos_bank: Sequence[np.ndarray],
    fs_hz: float,
    tau_s: float,
    stats: NormalizerStats | None = None,
) -> tuple[np.ndarray, NormalizerStats | None]:
    """
    Full batch preprocessing: CAR -> bandpower -> log -> normalize.

    Inputs: x (T, C), sos_bank (B), fs_hz, tau_s, optional stats (1, B, C).
    Output: features (T, B, C) and stats (1, B, C) if computed.
    """
    x_car = common_average_reference(x)
    feat = compute_log_power_features(x_car, sos_bank, fs_hz, tau_s)
    if stats is None:
        stats = fit_normalizer(feat)
    feat = apply_normalizer(feat, stats)
    return feat, stats


def build_electrode_mask(channels_coords: np.ndarray) -> np.ndarray:
    """Build electrode mask grid: input coords (C,2) -> output (H,W)."""
    if channels_coords.shape != (N_CHANNELS, 2):
        raise ValueError(
            f"channels_coords must have shape ({N_CHANNELS}, 2), got {channels_coords.shape}"
        )
    mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    y_offset = 2
    x = channels_coords[:, 0].astype(int)
    y = channels_coords[:, 1].astype(int) - y_offset
    valid = (x >= 0) & (x < GRID_WIDTH) & (y >= 0) & (y < GRID_HEIGHT)
    mask[y[valid], x[valid]] = 1.0
    return mask


def spatialize_band_features(
    features: np.ndarray,
    channels_coords: np.ndarray,
) -> np.ndarray:
    """Map band features to grid: input (T,B,C) -> output (T,B,H,W)."""
    if features.ndim != 3:
        raise ValueError(f"features must have shape (T,B,C), got {features.shape}")
    if features.shape[2] != N_CHANNELS:
        raise ValueError(
            f"features last dim must be {N_CHANNELS}, got {features.shape[2]}"
        )
    if channels_coords.shape != (N_CHANNELS, 2):
        raise ValueError(
            f"channels_coords must have shape ({N_CHANNELS}, 2), got {channels_coords.shape}"
        )
    y_offset = 2
    x = channels_coords[:, 0].astype(int)
    y = channels_coords[:, 1].astype(int) - y_offset
    valid = (x >= 0) & (x < GRID_WIDTH) & (y >= 0) & (y < GRID_HEIGHT)
    channel_idx = np.where(valid)[0]
    x = x[valid]
    y = y[valid]

    t, b, _ = features.shape
    grid = np.zeros((t, b, GRID_HEIGHT, GRID_WIDTH), dtype=features.dtype)
    grid[:, :, y, x] = features[:, :, channel_idx]
    return grid


def append_mask_channel(
    spatial_features: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Append mask: input (T,B,H,W) + (H,W) -> output (T,B+1,H,W)."""
    if spatial_features.ndim != 4:
        raise ValueError(
            f"spatial_features must have shape (T,B,H,W), got {spatial_features.shape}"
        )
    if mask.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(f"mask must have shape ({GRID_HEIGHT}, {GRID_WIDTH})")
    t = spatial_features.shape[0]
    mask_channel = np.broadcast_to(mask, (t, 1, GRID_HEIGHT, GRID_WIDTH))
    return np.concatenate([spatial_features, mask_channel], axis=1)


def pool_spatial_features(
    spatial_features: np.ndarray,
    kernel_size: int = 2,
    mask_channel_index: int | None = None,
) -> np.ndarray:
    """Downsample spatial grid by average pooling (mask uses max if provided)."""
    if spatial_features.ndim != 4:
        raise ValueError(
            f"spatial_features must have shape (T,C,H,W), got {spatial_features.shape}"
        )
    t, c, h, w = spatial_features.shape
    h_pad = (kernel_size - (h % kernel_size)) % kernel_size
    w_pad = (kernel_size - (w % kernel_size)) % kernel_size
    if h_pad or w_pad:
        x = np.pad(
            spatial_features,
            ((0, 0), (0, 0), (0, h_pad), (0, w_pad)),
            mode="constant",
        )
    else:
        x = spatial_features
    h_padded, w_padded = x.shape[2], x.shape[3]
    new_h = h_padded // kernel_size
    new_w = w_padded // kernel_size

    x = x.reshape(t, c, new_h, kernel_size, new_w, kernel_size)
    pooled = x.mean(axis=(3, 5))

    if mask_channel_index is not None:
        mask = x[:, mask_channel_index : mask_channel_index + 1, :, :]
        mask = mask.reshape(t, 1, new_h, kernel_size, new_w, kernel_size)
        pooled_mask = mask.max(axis=(3, 5))
        pooled[:, mask_channel_index : mask_channel_index + 1, :, :] = pooled_mask

    return pooled


def preprocess_spatial_features(
    x: np.ndarray,
    channels_coords: np.ndarray,
    sos_bank: Sequence[np.ndarray],
    fs_hz: float,
    tau_s: float,
    stats: NormalizerStats | None = None,
    pool_kernel: int = 2,
    append_mask: bool = True,
) -> tuple[np.ndarray, NormalizerStats | None]:
    """
    Full preprocessing to spatial grids.

    Inputs: x (T,C), coords (C,2), sos_bank (B), fs_hz, tau_s.
    Output: features (T,C2,H2,W2), stats (1,B,C) if computed.
    """
    band_feat, stats = preprocess_features(x, sos_bank, fs_hz, tau_s, stats=stats)
    spatial = spatialize_band_features(band_feat, channels_coords)
    if append_mask:
        mask = build_electrode_mask(channels_coords)
        spatial = append_mask_channel(spatial, mask)
        mask_idx = spatial.shape[1] - 1
    else:
        mask_idx = None
    spatial = pool_spatial_features(spatial, kernel_size=pool_kernel, mask_channel_index=mask_idx)
    return spatial, stats
