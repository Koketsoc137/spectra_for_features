"""
Drop-in replacements for the hot path of feature_tpcf.py.

Changes vs. original:
  * pair counting via chunked float32 GEMM instead of BallTree
  * bins in cosine distance directly (no euclidean round-trip, no sqrt)
  * no matplotlib import
  * no redundant re-normalisation
  * cos_dist_range lower bound validated instead of silently producing NaN
"""

from __future__ import annotations

import numpy as np
from scipy.special import betainc
from sklearn.decomposition import PCA


def rr_fraction(cos_edges, k):
    """Expected pair fraction per cosine-distance bin, uniform on S^(k-1)."""
    a = 0.5 * (k - 1)
    u = np.clip(1.0 - 0.5 * np.asarray(cos_edges, float), 0.0, 1.0)
    return -np.diff(betainc(a, a, u))


def _l2_normalize(X, eps=1e-12):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def _pair_counts(Z, cos_edges, block=2048):
    """
    Ordered-distinct-pair counts per cosine-distance bin.

    Z          : (n, k) float32, rows already L2-normalised
    cos_edges  : (nbins + 1,) increasing edges in cosine distance
    Self-pairs sit at d_cos = 0; they are excluded as long as
    cos_edges[0] > 0, which is enforced by the caller.
    """
    n = Z.shape[0]
    counts = np.zeros(len(cos_edges) - 1, dtype=np.int64)
    for s in range(0, n, block):
        S = Z[s:s + block] @ Z.T          # cosine similarity
        np.subtract(1.0, S, out=S)        # -> cosine distance, in place
        counts += np.histogram(S, bins=cos_edges)[0]
    return counts


def two_point_cos(data, cos_edges, normalized=False, block=2048):
    """
    Two-point correlation on the unit sphere, binned in cosine distance.
    Returns (nbins,) xi with NaN where the expected pair fraction is zero.
    """
    Z = np.asarray(data, dtype=np.float32)
    if Z.ndim != 2:
        raise ValueError(f"data must be 2-D (N, D); got {Z.shape}")
    if not normalized:
        Z = _l2_normalize(Z)
    Z = np.ascontiguousarray(Z, dtype=np.float32)

    n, k = Z.shape
    edges = np.asarray(cos_edges, dtype=np.float32)
    counts = _pair_counts(Z, edges, block=block)

    frac = rr_fraction(cos_edges, k)
    zero = frac <= 0
    corr = counts / (n * (n - 1.0) * np.where(zero, 1.0, frac)) - 1.0
    corr[zero] = np.nan
    return corr


def cosine_tpcf_score(
    features,
    n_components: int = 15,
    n_bins: int = 10,
    cos_dist_range: tuple[float, float] = (0.05, 1.0),
    random_state: int = 42,
) -> float:
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"features must be 2-D (N, D); got {X.shape}")

    lo, hi = cos_dist_range
    if not (0.0 < lo < hi):
        raise ValueError(
            f"cos_dist_range must satisfy 0 < lo < hi; got {cos_dist_range}. "
            "A lower edge of 0 gives log10(0) = -inf and all-NaN bins."
        )

    N, D = X.shape
    k = min(n_components, D, N)
    Z = PCA(n_components=k, random_state=random_state).fit_transform(X)

    cos_edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    return float(np.nansum(two_point_cos(_l2_normalize(Z), cos_edges, normalized=True)))


def _as_2d_array(representations) -> np.ndarray:
    """Coerce list / ndarray / list-of-tensors into a float (N, D) array."""
    if isinstance(representations, np.ndarray):
        return representations.astype(float, copy=False)
    return np.stack([np.asarray(a) for a in representations]).astype(float, copy=False)
