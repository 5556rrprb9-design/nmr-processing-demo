"""Simulate 1D 1H NMR free induction decays (FIDs) of a simple metabolite mixture.

Every spectrum is built from known peaks, so each processing step can be checked
against the truth. Some spectra get deliberate problems (broad lines, low signal,
a rolling baseline) so the quality control has something to catch.
"""
import numpy as np

SPECTROMETER_MHZ = 600.0      # 1H frequency
SWEEP_WIDTH_PPM = 12.0        # spectral window
CARRIER_PPM = 4.7             # centre of the window (water)
N_POINTS = 16384              # complex points acquired

# metabolite: list of (ppm, relative intensity) for each line, J couplings written out as lines
METABOLITES = {
    "TSP (reference)": [(0.00, 1.0)],
    "lactate": [(1.322, 1.0), (1.334, 1.0), (4.095, 0.25), (4.107, 0.75), (4.118, 0.75), (4.130, 0.25)],
    "alanine": [(1.470, 0.5), (1.482, 0.5)],
    "acetate": [(1.920, 1.0)],
    "citrate": [(2.525, 0.5), (2.551, 0.5), (2.664, 0.5), (2.690, 0.5)],
    "creatinine": [(3.040, 1.0), (4.050, 0.67)],
    "glucose (alpha H1)": [(5.217, 0.5), (5.229, 0.5)],
    "formate": [(8.450, 1.0)],
}

# concentrations (arbitrary units) per group; group B has more lactate and less citrate
GROUP_MEANS = {
    "A": {"TSP (reference)": 1.0, "lactate": 1.0, "alanine": 0.5, "acetate": 0.6,
          "citrate": 1.2, "creatinine": 1.5, "glucose (alpha H1)": 0.8, "formate": 0.3},
    "B": {"TSP (reference)": 1.0, "lactate": 1.8, "alanine": 0.5, "acetate": 0.6,
          "citrate": 0.6, "creatinine": 1.5, "glucose (alpha H1)": 0.8, "formate": 0.3},
}


def axis_info():
    sw_hz = SWEEP_WIDTH_PPM * SPECTROMETER_MHZ
    dwell = 1.0 / sw_hz
    t = np.arange(N_POINTS) * dwell
    return sw_hz, t


def simulate_fid(concentrations, linewidth_hz=1.0, noise=0.02, baseline_hump=0.0,
                 ph0_deg=0.0, ph1_deg=0.0, rng=None):
    """Return a complex FID for the given metabolite concentrations."""
    rng = np.random.default_rng() if rng is None else rng
    _, t = axis_info()
    t2 = 1.0 / (np.pi * linewidth_hz)                      # Lorentzian FWHM = 1/(pi*T2)
    fid = np.zeros(N_POINTS, dtype=complex)
    for name, lines in METABOLITES.items():
        conc = concentrations.get(name, 0.0)
        for ppm, rel in lines:
            f_hz = (ppm - CARRIER_PPM) * SPECTROMETER_MHZ
            fid += conc * rel * np.exp(2j * np.pi * f_hz * t) * np.exp(-t / t2)
    if baseline_hump > 0:                                  # very broad component = rolling baseline
        f_hz = (5.0 - CARRIER_PPM) * SPECTROMETER_MHZ      # (e.g. macromolecules or distorted first FID points)
        fid += baseline_hump * np.exp(2j * np.pi * f_hz * t) * np.exp(-t / 0.0002)
    fid += noise * (rng.standard_normal(N_POINTS) + 1j * rng.standard_normal(N_POINTS))
    # phase errors: constant (ph0) plus a first-order term that acts like a small time delay
    sw_hz, _ = axis_info()
    delay = (ph1_deg / 360.0) / sw_hz
    fid = fid * np.exp(1j * np.deg2rad(ph0_deg))
    fid = np.interp(t - delay, t, fid.real) + 1j * np.interp(t - delay, t, fid.imag)
    return fid


def simulate_dataset(n_per_group=20, n_bad=6, seed=7):
    """Two groups of spectra; n_bad of them get a deliberate quality problem."""
    rng = np.random.default_rng(seed)
    records = []
    for group in ("A", "B"):
        for i in range(n_per_group):
            conc = {k: v * rng.normal(1.0, 0.10) for k, v in GROUP_MEANS[group].items()}
            conc["TSP (reference)"] = 1.0
            records.append(dict(sample=f"{group}{i+1:02d}", group=group, conc=conc,
                                linewidth_hz=rng.uniform(0.8, 1.2), noise=0.02,
                                baseline_hump=0.0, problem="none"))
    bad_idx = rng.choice(len(records), size=n_bad, replace=False)
    problems = ["broad lines (poor shimming)", "low signal-to-noise", "rolling baseline"]
    for k, idx in enumerate(bad_idx):
        p = problems[k % 3]
        records[idx]["problem"] = p
        if p.startswith("broad"):
            records[idx]["linewidth_hz"] = rng.uniform(3.0, 5.0)
        elif p.startswith("low"):
            records[idx]["noise"] = 0.3
        else:
            records[idx]["baseline_hump"] = 40.0
    for r in records:
        r["fid"] = simulate_fid(r["conc"], r["linewidth_hz"], r["noise"], r["baseline_hump"],
                                ph0_deg=rng.uniform(-60, 60), ph1_deg=rng.uniform(-30, 30), rng=rng)
    return records
