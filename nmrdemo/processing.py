"""Processing steps: apodisation, zero filling, Fourier transform, phasing,
referencing, baseline correction and peak picking."""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks

from .simulate import SPECTROMETER_MHZ, SWEEP_WIDTH_PPM, CARRIER_PPM, axis_info


def apodise(fid, lb_hz=0.3):
    """Exponential line broadening: trades a little resolution for less noise."""
    _, t = axis_info()
    return fid * np.exp(-np.pi * lb_hz * t)


def zero_fill(fid, factor=2):
    """Append zeros to the FID; gives a finer digital resolution in the spectrum."""
    return np.concatenate([fid, np.zeros(len(fid) * (factor - 1), dtype=complex)])


def fourier_transform(fid):
    """FID -> spectrum, with a ppm axis running from high to low ppm (NMR convention)."""
    fid = fid.copy()
    fid[0] *= 0.5                                   # first-point correction against a baseline offset
    spec = np.fft.fftshift(np.fft.fft(fid))
    n = len(spec)
    ppm = CARRIER_PPM + np.linspace(SWEEP_WIDTH_PPM / 2, -SWEEP_WIDTH_PPM / 2, n, endpoint=False)
    return ppm, spec[::-1]


def apply_phase(spec, ph0_deg, ph1_deg):
    """Zero-order (ph0) and first-order (ph1) phase correction."""
    n = len(spec)
    x = np.arange(n) / n
    return spec * np.exp(1j * np.deg2rad(ph0_deg + ph1_deg * x))


def _entropy_cost(phases, spec):
    """ACME cost (Chen et al., J. Magn. Reson. 2002): entropy of the first
    derivative of the real part, plus a penalty for negative intensity."""
    real = apply_phase(spec, *phases).real
    d = np.abs(np.diff(real))
    h = d / d.sum()
    entropy = -np.sum(h * np.log(h + 1e-12))
    neg = real[real < 0]
    penalty = 1000 * np.sum(neg ** 2) / np.sum(real ** 2)
    return entropy + penalty


def auto_phase(spec):
    """Find ph0 and ph1 automatically. A coarse grid for ph0 avoids local minima."""
    best = None
    for start in range(-180, 180, 30):
        res = minimize(_entropy_cost, x0=[start, 0.0], args=(spec,), method="Nelder-Mead",
                       options={"xatol": 0.1, "fatol": 1e-6, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    ph0, ph1 = best.x
    return apply_phase(spec, ph0, ph1), (ph0 % 360, ph1)


def reference_to_tsp(ppm, spec, window=0.3):
    """Shift the ppm axis so the tallest peak near 0 ppm (TSP/DSS) sits at exactly 0.00 ppm."""
    mask = np.abs(ppm) < window
    peak_ppm = ppm[mask][np.argmax(spec.real[mask])]
    return ppm - peak_ppm, peak_ppm


def baseline_correct(ppm, real, order=4, n_iter=30):
    """Iterative polynomial baseline: fit, keep only points at or below the fit
    (signal-free regions), refit. Peaks never pull the baseline up."""
    x = (ppm - ppm.mean()) / np.ptp(ppm)
    y = real.copy()
    base = np.zeros_like(real)
    for _ in range(n_iter):
        coef = np.polyfit(x, y, order)
        base = np.polyval(coef, x)
        y = np.minimum(y, base)
    return real - base, base


def noise_std(ppm, real, region=(9.5, 10.5)):
    """Noise level from a region that contains no metabolite signals."""
    mask = (ppm > region[0]) & (ppm < region[1])
    return np.std(real[mask])


def pick_peaks(ppm, real, snr_threshold=10, exclude=((4.6, 4.9),)):
    """Peaks taller than snr_threshold x noise, outside the excluded regions (water)."""
    sigma = noise_std(ppm, real)
    # prominence (height above the surrounding signal) ignores slow baseline ripples
    idx, props = find_peaks(real, height=snr_threshold * sigma,
                            prominence=snr_threshold * sigma, distance=5)
    keep = np.ones(len(idx), bool)
    for lo, hi in exclude:
        keep &= ~((ppm[idx] > lo) & (ppm[idx] < hi))
    return ppm[idx][keep], props["peak_heights"][keep]


def process(fid):
    """Full pipeline for one FID. Returns ppm axis, baseline-corrected real spectrum,
    the phased spectrum before baseline correction, and a log of what was applied."""
    fid = zero_fill(apodise(fid))
    ppm, spec = fourier_transform(fid)
    spec, (ph0, ph1) = auto_phase(spec)
    ppm, shift = reference_to_tsp(ppm, spec)
    real, base = baseline_correct(ppm, spec.real)
    return ppm, real, spec.real, dict(ph0_deg=ph0, ph1_deg=ph1, reference_shift_ppm=shift)
