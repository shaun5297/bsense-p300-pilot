"""ERP feature path extracted from the existing dog-controller runtime."""
import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

ArtifactError = ValueError
SFREQ = 250.0
OFFSET = -0.2
WINDOW = 1.2
BANDPASS = (0.5, 20.0)

def preprocess_window(
    window: np.ndarray,
    sfreq: float,
    bandpass_hz: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(window, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ArtifactError(f"期望窗口形状为 (2, samples)，实际为 {values.shape}。")
    centered = values - np.median(values, axis=1, keepdims=True)
    low, high = bandpass_hz
    sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, centered, axis=1)


def signal_quality(processed: np.ndarray) -> tuple[bool, str]:
    if not np.isfinite(processed).all():
        return False, "non_finite"
    channel_std = processed.std(axis=1)
    if np.any(channel_std < 1e-8):
        return False, "flat_channel"
    standardized = processed / np.maximum(channel_std[:, None], 1e-8)
    if float(np.max(np.abs(standardized))) > 20.0:
        return False, "extreme_transient"
    return True, "ok"


def erp_features(
    processed: np.ndarray,
    sfreq: float,
    baseline_seconds: float = 0.2,
    output_sfreq: float = 50.0,
) -> tuple[np.ndarray, list[str]]:
    baseline_samples = max(1, int(round(baseline_seconds * sfreq)))
    corrected = processed - processed[:, :baseline_samples].mean(
        axis=1, keepdims=True
    )
    output_samples = int(round(corrected.shape[1] * output_sfreq / sfreq))
    downsampled = resample(corrected, output_samples, axis=1)
    names = [
        f"ch{channel + 1}_erp_t{sample / output_sfreq - baseline_seconds:.3f}"
        for channel in range(2)
        for sample in range(output_samples)
    ]
    return downsampled.reshape(-1).astype(np.float64), names
