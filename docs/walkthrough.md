# Walkthrough: what each step does and why

Read this once before an interview. Each section matches one function in `nmrdemo/`.

## 1. Simulation (`simulate.py`)
An FID is a sum of decaying complex exponentials, one per spectral line: amplitude × exp(2πi·f·t) × exp(−t/T2). The frequency f comes from the chemical shift (ppm − carrier) × 600 MHz, and the linewidth is 1/(π·T2). I add random zero- and first-order phase errors to every spectrum and Gaussian noise. Six spectra get a problem: broad lines (as with poor shimming), ten times more noise, or a very broad component that gives a rolling baseline.

## 2. Apodisation, zero filling, Fourier transform (`processing.py`)
- Exponential multiplication with 0.3 Hz line broadening lowers the noise at the end of the FID. The cost is 0.3 Hz extra linewidth, which is why the measured TSP linewidth is about 1.3 Hz for a true 1.0 Hz.
- Zero filling doubles the number of points and gives a finer digital resolution (0.22 Hz per point).
- The first FID point is halved before the FT. Without this, the spectrum gets a constant offset.

## 3. Phase correction
After the FT every peak is a mix of absorption and dispersion (top panel of figure 1). ph0 rotates all peaks equally, ph1 changes linearly across the spectrum. `auto_phase` searches ph0 and ph1 that minimise the entropy of the first derivative of the real spectrum and penalises negative intensity (ACME method, Chen et al. 2002). I start the search from 12 ph0 values to avoid local minima.

## 4. Referencing
The tallest point within ±0.3 ppm of zero is TSP; the axis is shifted to put it at exactly 0.00 ppm. In real samples DSS is often used instead of TSP because TSP is pH sensitive.

## 5. Baseline correction
A 4th-order polynomial is fitted, then every point above the fit is replaced by the fit and the fit is repeated. After 30 rounds the polynomial follows only the signal-free regions, so peaks do not pull it up.

## 6. Peak picking
`scipy.signal.find_peaks` with a minimum height and prominence of 10 × the noise standard deviation (measured at 9.5 to 10.5 ppm, where there is no signal). Prominence ignores slow baseline ripples. The water region (4.6 to 4.9 ppm) is excluded. Result: all 19 lines found in every spectrum.

## 7. Quality control (`qc.py`)
- **TSP linewidth** (full width at half maximum, interpolated): above 2.0 Hz means poor shimming.
- **Signal-to-noise**: TSP height / noise standard deviation; below 300 fails.
- **Baseline offset**: mean intensity at 5.6 to 7.8 ppm before baseline correction, relative to TSP; above 0.002 fails. Noise averages to about zero there, a rolling baseline does not.
The thresholds sit between the good and the damaged spectra in figure 2. On real data I would set them from the facility's reference samples.

## 8. Chemometrics (`analysis.py`)
- **Bucketing**: integrate the spectrum in 0.04 ppm bins, excluding water. Small shift differences between samples then do not matter.
- **PQN normalisation** (Dieterle et al., Anal. Chem. 2006): corrects for dilution differences between samples, standard for biofluids.
- **Pareto scaling + PCA**: Pareto scaling (divide by √SD) keeps large peaks important without letting them dominate. PC1 explains 72 % of the variance and separates the two groups; the loadings show lactate up and citrate down in group B, which is exactly what I put into the simulation.

## 9. Machine-learning check
A logistic regression learns pass/fail from the three QC features, with 5-fold cross-validation so it is always tested on spectra it has not seen. It gets 38 of 40 right, the thresholds get 40 of 40. With only 6 bad examples a model cannot beat clear thresholds; the honest conclusion is that ML pays off on large data sets and on artefacts that no single number captures.

## Questions Professor Orts might ask
- *Why entropy minimisation for phasing?* It needs no manual peak selection and works on crowded spectra; the negativity penalty prevents the inverted solution.
- *What would change with real Bruker data?* Read with nmrglue, remove the digital filter (group delay) before the FT, suppress or cut the water signal, and possibly align peaks (e.g. icoshift) before bucketing.
- *How would you scale this for a facility?* Run it automatically on every new data set, write the QC table to a database, and alert users when a spectrum fails, for example from the instrument's own reference sample each morning.
- *How did you build it?* With AI assistance, then I went through and checked each step myself. My own data evaluation code from the doctorate (FT-IR, Raman, QCM, AFM) follows the same logic of reproducible, automated evaluation.
