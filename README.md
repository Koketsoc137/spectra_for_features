# spectra_for_features

Measuring the structure of learned galaxy-image representations with the two-point correlation function (TPCF), intrinsic dimension, anisotropy and clustering metrics.

## Metrics over training

![Test-set representation metrics per epoch](notebooks/Test_metrics_grid.png)

Good representations tend to have lower-dimensional, anisotropic manifolds: as KNN accuracy improves over training, the intrinsic dimension falls and the variance concentrates in a small number of directions.

The figure is produced by [notebooks/Model_TPCF_scores.ipynb](notebooks/Model_TPCF_scores.ipynb).
