"""Run the whole demo: simulate 40 spectra, process them, run QC, PCA and the ML check.
Writes figures and tables to results/."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nmrdemo.simulate import simulate_dataset, METABOLITES
from nmrdemo.processing import process, pick_peaks, fourier_transform, zero_fill, apodise
from nmrdemo.qc import qc_report
from nmrdemo.analysis import bin_spectrum, pqn_normalise, pca_pareto, ml_qc_check

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

records = simulate_dataset()
rows, spectra = [], {}
for r in records:
    ppm, real, raw, log = process(r["fid"])
    qc = qc_report(ppm, real, raw)
    peaks, _ = pick_peaks(ppm, real)
    spectra[r["sample"]] = (ppm, real)
    rows.append(dict(sample=r["sample"], group=r["group"], true_problem=r["problem"],
                     n_peaks=len(peaks), **log, **qc))
    print(f'{r["sample"]}: QC {"pass" if qc["qc_pass"] else "FAIL (" + qc["qc_flags"] + ")"}')

table = pd.DataFrame(rows)
table.round(4).to_csv(OUT / "qc_report.csv", index=False)

# ---------- figure 1: raw vs processed spectrum ----------
first = records[0]
ppm_raw, spec_raw = fourier_transform(zero_fill(apodise(first["fid"])))
ppm, real = spectra[first["sample"]]
fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
ax[0].plot(ppm_raw, spec_raw.real, lw=0.6, color="0.4")
ax[0].set_title(f'{first["sample"]}: after Fourier transform (unphased)')
ax[1].plot(ppm, real, lw=0.6, color="k")
peaks, heights = pick_peaks(ppm, real)
ax[1].plot(peaks, heights, "v", ms=4, color="tab:red", label="picked peaks")
ax[1].set_title("after phasing, referencing to TSP and baseline correction")
ax[1].legend(frameon=False)
ax[1].set_xlim(9, -0.3)
ax[1].set_xlabel("chemical shift / ppm")
fig.tight_layout(); fig.savefig(OUT / "fig1_processing.png", dpi=200); plt.close(fig)

# ---------- figure 2: QC ----------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
colors = table["qc_pass"].map({True: "tab:blue", False: "tab:red"})
ax[0].scatter(table["linewidth_hz"], table["snr"], c=colors, s=18)
ax[0].set_xlabel("TSP linewidth / Hz"); ax[0].set_ylabel("signal-to-noise"); ax[0].set_yscale("log")
ax[0].set_title("QC: blue = pass, red = fail")
ax[1].bar(range(len(table)), table["baseline"], color=colors)
ax[1].set_xlabel("spectrum"); ax[1].set_ylabel("baseline offset (rel. to TSP)"); ax[1].set_yscale("log")
ax[1].set_title("baseline flatness")
fig.tight_layout(); fig.savefig(OUT / "fig2_qc.png", dpi=200); plt.close(fig)

# ---------- PCA on the spectra that passed QC ----------
good = table[table["qc_pass"]]
centres = None
X = []
for s in good["sample"]:
    centres, v = bin_spectrum(*spectra[s])
    X.append(v)
X = pqn_normalise(np.array(X))
scores, loadings, evr = pca_pareto(X)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for g, c in (("A", "tab:blue"), ("B", "tab:orange")):
    m = (good["group"] == g).values
    ax[0].scatter(scores[m, 0], scores[m, 1], s=20, color=c, label=f"group {g}")
ax[0].set_xlabel(f"PC1 ({evr[0]*100:.0f} %)"); ax[0].set_ylabel(f"PC2 ({evr[1]*100:.0f} %)")
ax[0].legend(frameon=False); ax[0].set_title("PCA scores (spectra that passed QC)")
ax[1].plot(centres, loadings[0], color="k", lw=0.8)
ax[1].set_xlim(9, 0.5); ax[1].set_xlabel("bucket centre / ppm"); ax[1].set_ylabel("PC1 loading")
ax[1].set_title("PC1 loadings")
for name, ppm_mark, offset in (("lactate", 1.33, (-55, -10)), ("lactate", 4.11, (8, -10)),
                               ("citrate", 2.6, (12, -20))):
    i = np.argmin(abs(centres - ppm_mark))
    ax[1].annotate(name, (centres[i], loadings[0][i]), textcoords="offset points",
                   xytext=offset, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig3_pca.png", dpi=200); plt.close(fig)

# ---------- ML check of the QC ----------
truth = (table["true_problem"] != "none").astype(int).values
pred = ml_qc_check(table[["linewidth_hz", "snr", "baseline"]].values, truth)
rule = (~table["qc_pass"]).astype(int).values
summary = pd.DataFrame({
    "method": ["fixed thresholds", "logistic regression (5-fold CV)"],
    "correct": [(rule == truth).sum(), (pred == truth).sum()],
    "missed bad spectra": [((rule == 0) & (truth == 1)).sum(), ((pred == 0) & (truth == 1)).sum()],
    "false alarms": [((rule == 1) & (truth == 0)).sum(), ((pred == 1) & (truth == 0)).sum()],
    "total": [len(truth)] * 2,
})
summary.to_csv(OUT / "qc_method_comparison.csv", index=False)
print(summary.to_string(index=False))
print(f"PC1 explains {evr[0]*100:.0f} % of the variance")
