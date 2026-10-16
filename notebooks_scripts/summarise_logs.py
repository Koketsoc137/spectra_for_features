"""Summarise the BYOL / DINO / supervised training logs and write paper plots to plots/."""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

root = Path(__file__).resolve().parents[1]
logs, out = root / "logs", root / "plots"
out.mkdir(exist_ok=True)

# label-noise setting is not printed in the log, so map job id -> label here
SUP_LABELS = {"14003375": "30% noisy labels", "14009055": "clean labels"}
C_TRAIN, C_VAL = "#2a78d6", "#eb6834"

plt.style.use(["science", "no-latex"])
plt.rc("font", size=9)


def ssl_log(path):
    rows = re.findall(r"Epoch \[(\d+)/\d+\], Train loss: ([^,\s]+), Validation loss: (\S+)", path.read_text(errors="ignore"))
    return pd.DataFrame(rows, columns=["epoch", "train_loss", "val_loss"]).astype(float)


def sup_log(path):
    t = path.read_text(errors="ignore")
    tr = re.findall(r"Train Accuracy: (\S+)%", t)
    te = re.findall(r"Test Accuracy: (\S+)%", t)
    n = min(len(tr), len(te))
    df = pd.DataFrame({"epoch": range(n), "train_acc": tr[:n], "test_acc": te[:n]}).astype(float)
    loss = re.findall(r"Epoch \[\d+/\d+\], Loss: (\S+)", t)
    df["train_loss"] = pd.to_numeric(pd.Series([None] + loss))  # epoch 0 is before training
    return df


runs, summary = {}, []
for method in ["byol", "dino", "supervised"]:
    for path in sorted(logs.glob(f"{method}_galaxy10_*.log")):
        job = path.stem.split("_")[-1]
        t = path.read_text(errors="ignore")
        df = sup_log(path) if method == "supervised" else ssl_log(path)
        if df.empty:
            continue
        runs[(method, job)] = df
        start = re.search(r"Job started on \S+ at (.+)", t)
        end = re.search(r"Job finished at (.+)", t)
        row = {"method": method, "job": job, "epochs": int(df.epoch.max()),
               "of": int(re.search(r"Epoch \[\d+/(\d+)\]", t).group(1)), "finished": bool(end),
               "hours": round((pd.to_datetime(end.group(1).replace("SAST ", "")) - pd.to_datetime(start.group(1).replace("SAST ", ""))).total_seconds() / 3600, 2) if end else None,
               "final_train_loss": df.train_loss.iloc[-1]}
        if method == "supervised":
            row |= {"final_train_acc": df.train_acc.iloc[-1], "final_test_acc": df.test_acc.iloc[-1],
                    "best_test_acc": df.test_acc.max(), "best_test_acc_epoch": int(df.epoch[df.test_acc.idxmax()])}
        else:
            row |= {"final_val_loss": df.val_loss.iloc[-1], "min_val_loss": df.val_loss.min(),
                    "min_val_loss_epoch": int(df.epoch[df.val_loss.idxmin()])}
        summary.append(row)

summary = pd.DataFrame(summary)
summary.to_csv(out / "training_summary.csv", index=False)
print(summary.to_string(index=False))

# SSL loss: latest run of each method, one panel each (different loss scales)
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
for ax, method in zip(axes, ["byol", "dino"]):
    job = max(j for m, j in runs if m == method)
    df = runs[(method, job)]
    ax.plot(df.epoch, df.train_loss, color=C_TRAIN, lw=1.5, label="Train")
    ax.plot(df.epoch, df.val_loss, color=C_VAL, lw=1.5, label="Validation")
    ax.set(title=method.upper(), xlabel="Epoch", ylabel="Loss")
axes[0].legend()
fig.tight_layout()
fig.savefig(out / "ssl_loss.pdf")
fig.savefig(out / "ssl_loss.png", dpi=300)

# Supervised: loss and accuracy, line style per run
fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(7, 2.6))
for (method, job), ls in zip([k for k in runs if k[0] == "supervised"], ["--", "-", ":", "-."]):
    df, name = runs[(method, job)], SUP_LABELS.get(job, job)
    ax_l.plot(df.epoch, df.train_loss, color=C_TRAIN, ls=ls, lw=1.5, label=f"Train ({name})")
    ax_a.plot(df.epoch, df.train_acc, color=C_TRAIN, ls=ls, lw=1.5, label=f"Train ({name})")
    ax_a.plot(df.epoch, df.test_acc, color=C_VAL, ls=ls, lw=1.5, label=f"Test ({name})")
ax_l.set(xlabel="Epoch", ylabel="Cross-entropy loss")
ax_a.set(xlabel="Epoch", ylabel="Accuracy (%)")
ax_l.legend()
ax_a.legend()
fig.tight_layout()
fig.savefig(out / "supervised_training.pdf")
fig.savefig(out / "supervised_training.png", dpi=300)
