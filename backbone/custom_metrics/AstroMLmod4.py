"""
feature_tpcf.py
===============
TPCF *score* for deep feature embeddings, computed **only** via the
cosine-similarity two-point correlation function.

Pipeline (single value):
    features (N, D)
      -> random subsample (sub_sample fraction)          [bootstrap]
      -> PCA to the first `n_components` (default 15)     [avoid HD effects]
      -> L2-normalise onto the unit hypersphere           [cosine metric]
      -> 2PCF vs analytic sphere-uniform randoms          [scalar per bin]
      -> score = sum of xi over valid bins                [scalar]

`TPCF_score` repeats that over `Nbootstrap` subsamples and returns
(mean, std) of the score.

Why cosine + sphere randoms
---------------------------
For L2-normalised u, v:  ||u - v||^2 = 2 (1 - cos(u, v)) = 2 * d_cos, so
Euclidean distance on the unit sphere is a monotonic function of cosine
distance. We bin in cosine distance and convert edges via
d_euc = sqrt(2 * d_cos). Randoms are uniform ON THE SPHERE, and because that
distribution is known in closed form the RR and DR counts are never made --
only DD is counted, which is ~1% of the pair work of the Monte-Carlo version.

Public API
----------
    TPCF_score(representations, ...) -> (mean, std)
    cosine_tpcf_score(features, ...) -> float
"""

from __future__ import annotations

import random

import numpy as np
from joblib import Parallel, delayed
from scipy.special import betainc
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def rr_fraction(cos_edges, k):
    """
    Expected fraction of pairs per cosine-distance bin when at least one
    member is uniform on S^(k-1): (1 + cos)/2 ~ Beta(a, a), a = (k-1)/2.
        Mardia & Jupp, Directional Statistics, 2nd ed. (2000), sec. 9.3
        Watson, Statistics on Spheres (1983)
    By rotational invariance this holds for DR as well as RR, so the two are
    equal and the data distribution never enters.

    Distributions of Angles in Random Packing on Spheres@article{cai2013angles,
    author  = {Cai, T. Tony and Fan, Jianqing and Jiang, Tiefeng},
    title   = {Distributions of Angles in Random Packing on Spheres},
    journal = {Journal of Machine Learning Research},
    volume  = {14},
    number  = {57},
    pages   = {1837--1864},
    year    = {2013}
    }
    """
    a = 0.5 * (k - 1)
    u = np.clip(1.0 - 0.5 * np.asarray(cos_edges, float), 0.0, 1.0)
    return -np.diff(betainc(a, a, u))




def two_point(data, bins):
    """
    Two-point correlation function on the unit sphere.

    data : (n_samples, n_features), L2-normalised internally
    bins : (Nbins + 1,) EUCLIDEAN edges; d_cos = bins**2 / 2
    returns : (Nbins,) xi, NaN where the expected RR is zero
    """
    data = np.asarray(data, float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise ValueError("data should be 1D or 2D")
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    bins = np.asarray(bins, float)
    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    n_samples, n_features = data.shape

    # Ball tree: Omohundro, "Five Balltree Construction Algorithms",
    # TR-89-063, ICSI Berkeley (1989). Pruning degrades towards brute force
    # with dimension, hence the PCA cap at n_components=15.
    KDT_D = BallTree(data, metric="euclidean")
    DD = np.diff(KDT_D.two_point_correlation(data, bins, dualtree=True))

    # RR == DR in closed form, so Landy-Szalay (ApJ 412, 64, 1993) collapses
    # to the natural estimator; the sphere has no boundary to correct for.
    frac = rr_fraction(bins ** 2 / 2.0, n_features)
    zero = frac <= 0
    corr = DD / (n_samples * (n_samples - 1.0) * np.where(zero, 1.0, frac)) - 1.0
    corr[zero] = np.nan
    import matplotlib as plt
    #plt.plot(corr)
    #plt.shopw()
    return corr


def _l2_normalize(X, eps=1e-12):
    """Project rows onto the unit hypersphere."""
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def _as_2d_array(representations) -> np.ndarray:
    """Coerce list / ndarray / list-of-tensors into a float (N, D) array."""
    if isinstance(representations, np.ndarray):
        return representations.astype(float, copy=False)
    first = representations[0]
    if isinstance(first, list):
        return np.asarray(representations, dtype=float)
    return np.asarray([np.asarray(a).tolist() for a in representations], dtype=float)


# --------------------------------------------------------------------------
# Core: single cosine-TPCF score
# --------------------------------------------------------------------------


def cosine_tpcf_score(
    features,
    n_components: int = 15,
    n_bins: int = 100,
    cos_dist_range: tuple[float, float] = (0.0, 2.0),
    random_state: int = 42,
) -> float:
    """
    Cosine-metric 2PCF on the first `n_components` PCA components; returns
    the summed-xi score.

    cos_dist_range : d_cos = 1 - cos_sim. On a 15-D sphere uniform mass
        concentrates near the equator with width ~1/sqrt(k), so bins below
        ~0.03-0.05 are effectively empty; the default lower edge avoids that.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"features must be 2-D (N, D); got {X.shape}")

    N, D = X.shape
    k = min(n_components, D, N)

    Z = PCA(n_components=k, random_state=random_state).fit_transform(X)
    block = _l2_normalize(Z)

    lo, hi = cos_dist_range
    cos_edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    euc_edges = np.sqrt(2.0 * cos_edges)

    return float(np.nansum(two_point(block, euc_edges)))


# --------------------------------------------------------------------------
# Bootstrap orchestration
# --------------------------------------------------------------------------


def _single_bootstrap(i, representations, sub_sample, n_components,
                      n_bins, cos_dist_range, seed, verbose):
    rng = random.Random(seed + i)
    k = max(n_components, int(len(representations) * sub_sample))
    k = min(k, len(representations))
    indices = rng.sample(range(len(representations)), k)

    score = cosine_tpcf_score(
        representations[indices, :],
        n_components=n_components,
        n_bins=n_bins,
        cos_dist_range=cos_dist_range,
        random_state=seed + i,
    )
    if verbose:
        print(f"Bootstrap {i}: score {score:.3f}")
    return score


def TPCF_score(
    representations,
    epoch: int = 0,                 # accepted for drop-in compatibility (unused)
    sub_sample: float = 0.6,
    Nbootstrap: int = 50,
    n_components: int = 15,
    n_bins: int = 10,
    cos_dist_range: tuple[float, float] = (0.05, 1.0),
    verbose: bool = False,
    n_jobs: int = -1,
    seed: int = 0,
):
    """
    TPCF score over `Nbootstrap` random subsamples.

    Returns (mean, std), rounded to 2 dp and NaN-masked.
    """
    representations = _as_2d_array(representations)

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap)(
            i, representations, sub_sample, n_components,
            n_bins, cos_dist_range, seed, verbose
        )
        for i in range(Nbootstrap)
    )

    scores = np.ma.masked_invalid(scores)
    return (round(float(scores.mean()), 2),
            round(float(scores.std(ddof=1)), 2))


if __name__ == "__main__":
    # quick self-check: clustered features should score well above noise
    rng = np.random.default_rng(0)
    d, ncl = 128, 8
    centers = rng.standard_normal((ncl, d)) * 3.0
    lab = rng.integers(0, ncl, size=3000)
    clustered = centers[lab] + rng.standard_normal((3000, d))
    noise = rng.standard_normal((3000, d))

    print("clustered:", TPCF_score(clustered, Nbootstrap=10, n_jobs=1))
    print("noise    :", TPCF_score(noise,     Nbootstrap=10, n_jobs=1))