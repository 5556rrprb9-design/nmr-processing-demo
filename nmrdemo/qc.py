"""Quality control per spectrum: reference linewidth, signal-to-noise, baseline flatness."""
import numpy as np
from .processing import noise_std
from .simulate import SPECTROMETER_MHZ

# thresholds chosen for this simulated data set, see docs/walkthrough.md
MAX_LINEWIDTH_HZ = 2.0
MIN_SNR = 300
MAX_BASELINE = 0.002


def reference_linewidth_hz(ppm, real, window=0.05):
    """Full width at half maximum of the TSP peak, in Hz."""
    mask = np.abs(ppm) < window
    x, y = ppm[mask], real[mask]
    i = np.argmax(y)
    half = y[i] / 2
    # walk out from the maximum and interpolate the exact half-height crossing on each side
    r = i
    while y[r] > half:
        r += 1
    l = i
    while y[l] > half:
        l -= 1
    x_right = np.interp(half, [y[r], y[r - 1]], [x[r], x[r - 1]])
    x_left = np.interp(half, [y[l], y[l + 1]], [x[l], x[l + 1]])
    return abs(x_left - x_right) * SPECTROMETER_MHZ


def snr(ppm, real):
    """Height of the TSP peak divided by the noise standard deviation."""
    return real[np.abs(ppm) < 0.05].max() / noise_std(ppm, real)


def baseline_flatness(ppm, raw_real, region=(5.6, 7.8)):
    """Mean intensity of the phased but not yet baseline-corrected spectrum in a
    signal-free region, relative to the TSP peak. Noise averages out to about zero,
    a rolling baseline does not."""
    mask = (ppm > region[0]) & (ppm < region[1])
    return abs(np.mean(raw_real[mask])) / raw_real[np.abs(ppm) < 0.05].max()


def qc_report(ppm, real, raw_real):
    lw = reference_linewidth_hz(ppm, real)
    s = snr(ppm, real)
    b = baseline_flatness(ppm, raw_real)
    flags = []
    if lw > MAX_LINEWIDTH_HZ:
        flags.append("linewidth")
    if s < MIN_SNR:
        flags.append("SNR")
    if b > MAX_BASELINE:
        flags.append("baseline")
    return dict(linewidth_hz=lw, snr=s, baseline=b,
                qc_pass=not flags, qc_flags=";".join(flags) or "-")
