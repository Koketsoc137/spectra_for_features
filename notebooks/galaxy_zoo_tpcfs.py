"""
TPCF comparison analysis: 2PCF score vs Davies-Bouldin and vs clustering accuracy,
across all foundation-model representations written by gz_foundation_models.py
(<name>_full.parquet / <name>_class.parquet). Also computes ID, cosine similarity,
anisotropy and KNN accuracy, as in deep_features.ipynb.

Saves:
    results/foundation_model_scores.csv  - all metrics per model
    results/tpcf_vs_davies_data.pkl      - raw scores/errors/labels for the Davies-Bouldin plot
    results/tpcf_vs_accuracy_data.pkl    - raw scores/errors/labels for the accuracy plot
    results/tpcf_vs_davies_fig.pkl       - pickled matplotlib Figure (Davies-Bouldin)
    results/tpcf_vs_accuracy_fig.pkl     - pickled matplotlib Figure (accuracy)
    results/tpcf_vs_davies.png           - rendered image (Davies-Bouldin)
    results/tpcf_vs_accuracy.png         - rendered image (accuracy)
"""

import glob
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import davies_bouldin_score as davies

import backbone.visuals.VISUAL as viz
import backbone.custom_metrics.AstroMLmod3 as AstroMLmod
import backbone.custom_metrics.Intrinsic_dimension as id_module
import backbone.data_handle.Test as test

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

verbose = 1


def open_embeds(file_path):
    df = pd.read_parquet(file_path)
    rep = df.drop(columns=["label", "id"]).to_numpy(dtype=np.float32)
    return rep, df["label"].tolist(), df["id"].tolist()


def pickle_dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Load representations and compute metrics (one model at a time)
# ---------------------------------------------------------------------------

direc = "/idia/projects/camil/Koketso/galaxy_zoo_representations/"
models = [os.path.basename(p)[: -len("_full.parquet")] for p in sorted(glob.glob(direc + "*_full.parquet"))
          if os.path.exists(p.replace("_full.parquet", "_class.parquet"))]

rows = []
for name in models:
    rep, _, ids = open_embeds(direc + f"{name}_full.parquet")
    rep_c, label_c, ids_c = open_embeds(direc + f"{name}_class.parquet")
    print(f"{name}: rep={rep.shape}, rep_c={rep_c.shape}")

    tpcf = AstroMLmod.TPCF_score(rep)
    id_mean, id_std = id_module.id_score(rep)
    acc = test.clustering_accuracy((rep, ids), (label_c, ids_c))
    dav = davies(viz.pca(rep_c, n_components=15, verbose=verbose), label_c)
    knn_acc, knn_std = test.KNN_accuracy(rep_c, label_c)

    # exact mean/std of off-diagonal pairwise cosine similarity, without building the n x n matrix
    unit = rep / np.linalg.norm(rep, axis=1, keepdims=True)
    n = len(unit)
    unit_sum = unit.sum(axis=0, dtype=np.float64)
    gram = (unit.T @ unit).astype(np.float64)
    cos_mean = (unit_sum @ unit_sum - n) / (n * (n - 1))
    cos_std = np.sqrt(((gram ** 2).sum() - n) / (n * (n - 1)) - cos_mean ** 2)

    # anisotropy: top-1 explained variance ratio of the unit vectors
    anisotropy = IncrementalPCA(n_components=1).fit(unit).explained_variance_ratio_[0]

    print(f"TPCF- {name}: ", tpcf, dav, acc)

    rows.append({
        "model": name,
        "tpcf_mean": tpcf[0],
        "tpcf_std": tpcf[1],
        "id_mean": float(np.ravel(id_mean)[0]),
        "id_std": float(np.ravel(id_std)[0]),
        "avg_cosine_similarity": cos_mean,
        "cosine_similarity_std": cos_std,
        "anisotropy": anisotropy,
        "clustering_accuracy": acc,
        "davies_bouldin": dav,
        "knn_accuracy": knn_acc,
        "knn_accuracy_std": knn_std,
    })
    del rep, rep_c, unit

scores = pd.DataFrame(rows)
scores.to_csv(os.path.join(OUTPUT_DIR, "foundation_model_scores.csv"), index=False)
print(scores)

# ---------------------------------------------------------------------------
# Analysis 1: TPCF vs Davies-Bouldin
# ---------------------------------------------------------------------------

fig_davies, ax_davies = plt.subplots(figsize=(8, 6))
ax_davies.errorbar(scores["tpcf_mean"], scores["davies_bouldin"], xerr=scores["tpcf_std"],
                   fmt='o', capsize=4, markersize=8, ecolor='gray', zorder=2)

for x, y, name in zip(scores["tpcf_mean"], scores["davies_bouldin"], scores["model"]):
    ax_davies.annotate(name, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=10)

ax_davies.set_xlabel("2PCF Score")
ax_davies.set_ylabel("Davies-Bouldin Score")
ax_davies.set_title("2PCF Score vs Davies-Bouldin Score by Model")
ax_davies.grid(alpha=0.3)
plt.tight_layout()

davies_data = {
    "tpcf_means": scores["tpcf_mean"].tolist(),
    "tpcf_errors": scores["tpcf_std"].tolist(),
    "davies_scores": scores["davies_bouldin"].tolist(),
    "labels": scores["model"].tolist(),
}
pickle_dump(davies_data, os.path.join(OUTPUT_DIR, "tpcf_vs_davies_data.pkl"))
pickle_dump(fig_davies, os.path.join(OUTPUT_DIR, "tpcf_vs_davies_fig.pkl"))
fig_davies.savefig(os.path.join(OUTPUT_DIR, "tpcf_vs_davies.png"), dpi=300, bbox_inches="tight")
print(f"Saved: {os.path.join(OUTPUT_DIR, 'tpcf_vs_davies.png')}")


# ---------------------------------------------------------------------------
# Analysis 2: TPCF vs clustering accuracy
# ---------------------------------------------------------------------------

fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
ax_acc.errorbar(scores["tpcf_mean"], scores["clustering_accuracy"], xerr=scores["tpcf_std"],
                fmt='o', capsize=4, markersize=8, ecolor='gray', zorder=2)

for x, y, name in zip(scores["tpcf_mean"], scores["clustering_accuracy"], scores["model"]):
    ax_acc.annotate(name, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=10)

ax_acc.set_xlabel("2PCF Score")
ax_acc.set_ylabel("Clustering Accuracy")
ax_acc.set_title("2PCF Score vs Clustering Accuracy by Model")
ax_acc.grid(alpha=0.3)
plt.tight_layout()

# ---------------------------------------------------------------------------
# Save everything: raw results (pickle), figures (pickle), images (png)
# ---------------------------------------------------------------------------

accuracy_data = {
    "tpcf_means": scores["tpcf_mean"].tolist(),
    "tpcf_errors": scores["tpcf_std"].tolist(),
    "acc_scores": scores["clustering_accuracy"].tolist(),
    "labels": scores["model"].tolist(),
}
pickle_dump(accuracy_data, os.path.join(OUTPUT_DIR, "tpcf_vs_accuracy_data.pkl"))
pickle_dump(fig_acc, os.path.join(OUTPUT_DIR, "tpcf_vs_accuracy_fig.pkl"))
fig_acc.savefig(os.path.join(OUTPUT_DIR, "tpcf_vs_accuracy.png"), dpi=300, bbox_inches="tight")
print(f"Saved: {os.path.join(OUTPUT_DIR, 'tpcf_vs_accuracy.png')}")
