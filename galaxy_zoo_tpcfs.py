"""
TPCF comparison analysis: 2PCF score vs Davies-Bouldin and vs clustering accuracy,
across representation models (Zoobot, Dino, DinoHuge, ImNet, ResImNet).

Saves:
    results/tpcf_vs_davies_data.pkl      - raw scores/errors/labels for the Davies-Bouldin plot
    results/tpcf_vs_accuracy_data.pkl    - raw scores/errors/labels for the accuracy plot
    results/tpcf_vs_davies_fig.pkl       - pickled matplotlib Figure (Davies-Bouldin)
    results/tpcf_vs_accuracy_fig.pkl     - pickled matplotlib Figure (accuracy)
    results/tpcf_vs_davies.png           - rendered image (Davies-Bouldin)
    results/tpcf_vs_accuracy.png         - rendered image (accuracy)
"""

import os
import pickle
import random

import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score as davies

import backbone.VISUAL as viz
import backbone.AstroMLmod4 as AstroMLmod
import backbone.Test as test

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

verbose = 1


def save_embeds(embeddings, labels, ids, file_path="model_data.h5"):
    with h5py.File(file_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings, compression="gzip")
        f.create_dataset("labels", data=labels)
        f.create_dataset("ids", data=ids)
    print(f"Stored multiple arrays in {file_path}")


def open_embeds(file_path):
    with h5py.File(file_path, "r") as F:
        rep = np.array(F["embeddings"])
        labels = np.array(F["labels"])
        ids = np.array(F["ids"])
    return rep.astype(np.float32), labels.astype(str).tolist(), ids.tolist()


def pickle_dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Load representations
# ---------------------------------------------------------------------------

direc = "/idia/projects/camil/Koketso/galaxy_zoo_representations/"

zoobot_rep, zoobot_label, zoobot_ids = open_embeds(direc + "zoobot_reps.h5")
zoobot_rep_c, zoobot_label_c, zoobot_ids_c = open_embeds(direc + "zoobot_reps_c.h5")

dino_rep, dino_label, dino_ids = open_embeds(direc + "dinov3_convnext_base_reps.h5")
dino_rep_c, dino_label_c, dino_ids_c = open_embeds(direc + "dinov3_convnext_base_reps_c.h5")

dinov3_vith_rep, dinov3_vith_label, dinov3_vith_ids = open_embeds(direc + "dinov3_vith_reps.h5")
dinov3_vith_rep_c, dinov3_vith_label_c, dinov3_vith_ids_c = open_embeds(direc + "dinov3_vith_reps_c.h5")

imnet_rep, imnet_label, imnet_ids = open_embeds(direc + "imnet_convnext_base_reps.h5")
imnet_rep_c, imnet_label_c, imnet_ids_c = open_embeds(direc + "imnet_convnext_base_reps_c.h5")

res_imnet_rep, res_imnet_label, res_imnet_ids = open_embeds(direc + "imnet_resnet18_reps.h5")
res_imnet_rep_c, res_imnet_label_c, res_imnet_ids_c = open_embeds(direc + "imnet_resnet18_reps_c.h5")

# ---------------------------------------------------------------------------
# Analysis 1: TPCF vs Davies-Bouldin
# ---------------------------------------------------------------------------


    
davies_models = {
    "Zoobot":   (zoobot_rep_c,       zoobot_label_c),
    "Dino":     (dino_rep_c,         dino_label_c),
    "DinoHuge": (dinov3_vith_rep_c,  dinov3_vith_label_c),
    "ImNet":    (imnet_rep_c,        imnet_label_c),
    "ResImNet": (res_imnet_rep_c,    res_imnet_label_c),
}

tpcf_means_davies = []
tpcf_errors_davies = []
davies_scores = []
davies_labels = []

for name, (rep, lab) in davies_models.items():
    tpcf = AstroMLmod.TPCF_score(rep)
    dav = davies(viz.pca(rep, n_components=15, verbose=verbose), lab)

    print(f"TPCF- {name}: ", tpcf, dav)

    tpcf_means_davies.append(tpcf[0])
    tpcf_errors_davies.append(tpcf[1])
    davies_scores.append(dav)
    davies_labels.append(name)


fig_davies, ax_davies = plt.subplots(figsize=(8, 6))
ax_davies.errorbar(tpcf_means_davies, davies_scores, xerr=tpcf_errors_davies,
                    fmt='o', capsize=4, markersize=8, ecolor='gray', zorder=2)

for x, y, name in zip(tpcf_means_davies, davies_scores, davies_labels):
    ax_davies.annotate(name, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=10)

ax_davies.set_xlabel("2PCF Score")
ax_davies.set_ylabel("Davies-Bouldin Score")
ax_davies.set_title("2PCF Score vs Davies-Bouldin Score by Model")
ax_davies.grid(alpha=0.3)
plt.tight_layout()

davies_data = {
    "tpcf_means": tpcf_means_davies,
    "tpcf_errors": tpcf_errors_davies,
    "davies_scores": davies_scores,
    "labels": davies_labels,
}
pickle_dump(davies_data, os.path.join(OUTPUT_DIR, "3class_tpcf_vs_davies_data.pkl"))



davies_labels = ['Zoobot -  ConvNext', 'Dino - ConvNext', 'Dino - Huge', 'ImNet - ConvNext', 'ImNet - Res']
fig_davies, ax_davies = plt.subplots(figsize=(16, 12))
ax_davies.errorbar(tpcf_means_davies, davies_scores, xerr=tpcf_errors_davies,
                    fmt='o', capsize=4, markersize=16, ecolor='gray', zorder=2)

for x, y, name in zip(tpcf_means_davies, davies_scores, davies_labels):
    ax_davies.annotate(name, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=15)

ax_davies.set_xlabel("2PCF Score")
ax_davies.set_ylabel("Davies-Bouldin Score")
ax_davies.set_title("2PCF Score vs Davies-Bouldin Score by Model")
ax_davies.grid(alpha=0.3)
plt.tight_layout()
ax_davies.get_figure().savefig(os.path.join(OUTPUT_DIR, "3class_tpcf_vs_davies.png"))


pickle_dump(fig_davies, os.path.join(OUTPUT_DIR, "tpcf_vs_davies_fig.pkl"))
print(f"Saved: {os.path.join(OUTPUT_DIR, 'tpcf_vs_davies.png')}")


# ---------------------------------------------------------------------------
# Analysis 2: TPCF vs clustering accuracy
# ---------------------------------------------------------------------------

accuracy_models = {
    "Zoobot":   (zoobot_rep,       zoobot_ids,       zoobot_label_c,       zoobot_ids_c),
    "Dino":     (dino_rep,         dino_ids,         dino_label_c,         dino_ids_c),
    "DinoHuge": (dinov3_vith_rep,  dinov3_vith_ids,  dinov3_vith_label_c,  dinov3_vith_ids_c),
    "ImNet":    (imnet_rep,        imnet_ids,        imnet_label_c,        imnet_ids_c),
    "ResImNet": (res_imnet_rep,    res_imnet_ids,    res_imnet_label_c,    res_imnet_ids_c),
}

tpcf_means_acc = []
tpcf_errors_acc = []
acc_scores = []
acc_labels = []

for name, (rep, ids, lab_c, ids_c) in accuracy_models.items():

    tpcf = AstroMLmod.TPCF_score(rep)
    acc = test.clustering_accuracy((rep, ids), (lab_c, ids_c))
    print(f"TPCF- {name}: ", tpcf, acc)

    tpcf_means_acc.append(tpcf[0])
    tpcf_errors_acc.append(tpcf[1])
    acc_scores.append(acc)
    acc_labels.append(name)

fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
ax_acc.errorbar(tpcf_means_acc, acc_scores, xerr=tpcf_errors_acc,
                 fmt='o', capsize=4, markersize=8, ecolor='gray', zorder=2)

for x, y, name in zip(tpcf_means_acc, acc_scores, acc_labels):
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
    "tpcf_means": tpcf_means_acc,
    "tpcf_errors": tpcf_errors_acc,
    "acc_scores": acc_scores,
    "labels": acc_labels,
}
pickle_dump(accuracy_data, os.path.join(OUTPUT_DIR, "tpcf_vs_accuracy_data.pkl"))
pickle_dump(fig_acc, os.path.join(OUTPUT_DIR, "tpcf_vs_accuracy_fig.pkl"))
fig_acc.savefig(os.path.join(OUTPUT_DIR, "tpcf_vs_accuracy.png"), dpi=300, bbox_inches="tight")
print(f"Saved: {os.path.join(OUTPUT_DIR, 'tpcf_vs_accuracy.png')}")