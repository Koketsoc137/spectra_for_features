
"""Two-point correlation function (2PCF) score based on cosine similarity.

Embeddings are L2-normalized onto the unit hypersphere, so distance between
points reflects cosine similarity. The correlation function compares pair
counts in the data against a Gaussian background with the same mean/covariance:
a positive value at a given separation means the data is more clustered than
the background there. `TPCF_score` reduces this into a single mean/std score
over several bootstrapped, PCA-reduced subsamples.
"""

import random

import numpy as np
from scipy.special import betainc
from sklearn.neighbors import BallTree
from sklearn.utils import check_random_state

from ..visuals import Distributions as dist
from ..visuals import VISUAL as viz


def _unit_normalize(vectors):
    """L2-normalize each row so pairwise distance reflects cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def rr_fraction(bins, k):
    """Expected fraction of pairs per bin, uniform on S^(k-1) (analytic RR).

    Exact alternative to a sampled background: the chord distance between
    two independent uniform points on the (k-1)-sphere has a known
    Beta-distributed CDF, so the expected pair fraction is computed directly
    instead of estimated from noisy background samples.
    """
    a = 0.5 * (k - 1)
    u = np.clip(1.0 - 0.5 * np.asarray(bins, dtype=float), 0.0, 1.0)
    return -np.diff(betainc(a, a, u))


def _pair_counts(vectors, bins):
    """Count pairs within each distance bin using a BallTree."""
    tree = BallTree(vectors)
    return np.diff(tree.two_point_correlation(vectors, bins))


def two_point_correlation(data, bins, background=None, method="standard", min_pairs=10):
    """Two-point correlation of `data` against a comparison background.

    `data` (and `background`, if given) are unit-normalized before counting,
    so `bins` are chord distances on the unit sphere in [0, 2] (cosine
    similarity in [1, -1]).

    Parameters
    ----------
    data : array_like, shape = [n_samples, n_features]
    bins : array_like, shape = [n_bins + 1]
    background : array_like, shape = [n_background, n_features], optional
        Comparison sample, required for "standard"/"landy-szalay". Unused
        for "analytic".
    method : "standard", "landy-szalay", or "analytic"
        "analytic" skips the sampled `background` and instead compares
        against the exact pair fraction expected under a uniform
        distribution on the hypersphere (`rr_fraction`), removing the extra
        Monte Carlo noise a sampled background adds on top of the data's own
        sampling noise. `bins[0]` must be > 0 in that case, to exclude the
        n self-pairs every point forms with itself at distance 0.
    min_pairs : int
        Bins with fewer than this many background pairs (RR) are dropped,
        since a small RR makes the DD/RR ratio noisy and can blow up.

    Returns
    -------
    corr : ndarray, shape = [n_bins]
        Correlation estimate per bin; NaN where dropped.
    """
    if method not in ("standard", "landy-szalay", "analytic"):
        raise ValueError("method must be 'standard', 'landy-szalay', or 'analytic'")

    data = _unit_normalize(np.asarray(data))
    bins = np.asarray(bins, dtype=float)
    n = len(data)
    DD = _pair_counts(data, bins)

    if method == "analytic":
        if bins[0] <= 0:
            raise ValueError("bins[0] must be > 0 with method='analytic'")
        RR = rr_fraction(bins, data.shape[1]) * n * (n - 1)
        dropped = RR < min_pairs
        RR_safe = np.where(dropped, 1, RR)
        corr = DD / RR_safe - 1
    else:
        background = _unit_normalize(np.asarray(background))
        factor = len(background) / n
        RR = _pair_counts(background, bins)
        dropped = RR < min_pairs
        RR_safe = np.where(dropped, 1, RR)
        if method == "standard":
            corr = factor**2 * DD / RR_safe - 1
        else:  # landy-szalay
            DR = np.diff(BallTree(background).two_point_correlation(data, bins))
            corr = (factor**2 * DD - 2 * factor * DR + RR_safe) / RR_safe

    corr[dropped] = np.nan
    return corr


def _gaussian_background(data, n_points, seed=None):
    """Sample a Gaussian background matching the mean/covariance of `data`."""
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return dist.generate_gaussian_points(
        mean=mean, cov=cov, n_points=n_points, dimensions=data.shape[1], seed=seed
    )


def cosine_tpcf_score(data, bin_number=100, background_factor=10, method="analytic", random_state=None, min_pairs=10):

    data = np.asarray(data) - np.mean(data, axis=0)
    bins = np.linspace(1e-6 if method == "analytic" else 0, 2, bin_number)

    background = None
    if method != "analytic":
        rng = check_random_state(random_state)
        background = _gaussian_background(data, background_factor * len(data), seed=rng.randint(0, 10_000))

    corr = two_point_correlation(data, bins, background=background, method=method, min_pairs=min_pairs)
    return np.nanmean(corr)


def TPCF_score(representations, epoch=0, sub_sample=0.3, Nbootstrap=5, verbose=False, method="analytic"):
    representations = np.asarray(representations, dtype=float)
    sample_size = int(len(representations) * sub_sample)

    scores = []
    for _ in range(Nbootstrap):
        indices = random.sample(range(len(representations)), sample_size)
        reduced = viz.pca(representations[indices, :], n_components=15, verbose=verbose)
        scores.append(cosine_tpcf_score(reduced, method=method))

    scores = np.ma.masked_invalid(scores)
    return round(scores.mean(), 2), round(scores.std(ddof=1), 2)

