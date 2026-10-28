"""Chemometrics (binning + PCA) and a small machine-learning check of the QC."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def bin_spectrum(ppm, real, width=0.04, lo=0.5, hi=9.5, exclude=((4.6, 4.9),)):
    """Integrate the spectrum in buckets of `width` ppm (standard metabolomics bucketing)."""
    edges = np.arange(lo, hi + width, width)
    values = []
    centres = []
    for a, b in zip(edges[:-1], edges[1:]):
        c = (a + b) / 2
        if any(e0 <= c <= e1 for e0, e1 in exclude):
            continue
        mask = (ppm >= a) & (ppm < b)
        values.append(real[mask].sum())
        centres.append(c)
    return np.array(centres), np.array(values)


def pqn_normalise(X):
    """Probabilistic quotient normalisation (Dieterle et al., Anal. Chem. 2006):
    corrects for overall dilution differences between samples."""
    X = np.clip(X, 1e-12, None)
    X = X / X.sum(axis=1, keepdims=True)
    reference = np.median(X, axis=0)
    quotients = np.median(X / reference, axis=1, keepdims=True)
    return X / quotients


def pca_pareto(X, n_components=2):
    """PCA after mean-centring and Pareto scaling (divide by sqrt of std)."""
    Xc = X - X.mean(axis=0)
    Xs = Xc / np.sqrt(X.std(axis=0) + 1e-12)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xs)
    return scores, pca.components_, pca.explained_variance_ratio_


def ml_qc_check(features, labels, seed=0):
    """Logistic regression on the three QC features, 5-fold cross-validation.
    Returns the cross-validated predictions."""
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(model, np.log10(features), labels, cv=cv)
