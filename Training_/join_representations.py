import glob
import os

import pandas as pd
import yaml
from pathlib import Path

#same config as gz_foundation_models.py (CONFIG overrides, e.g. gz_foundation_models_radio.yaml)
config_path = os.environ.get("CONFIG", Path(__file__).with_name("gz_foundation_models.yaml"))
REP_DIR = yaml.safe_load(open(config_path))["paths"]["rep_dir"]

#join each model's train and val representations into one <name>_full.parquet
for train_path in sorted(glob.glob(os.path.join(REP_DIR, "*_train.parquet"))):
    name = os.path.basename(train_path)[: -len("_train.parquet")]
    val_path = os.path.join(REP_DIR, f"{name}_val.parquet")
    full_path = os.path.join(REP_DIR, f"{name}_full.parquet")

    if os.path.exists(full_path):
        print(f"[{name}] {full_path} already exists, skipping.")
        continue
    if not os.path.exists(val_path):
        print(f"[{name}] no val split yet, skipping.")
        continue

    df = pd.concat([pd.read_parquet(train_path), pd.read_parquet(val_path)], ignore_index=True)
    assert not df["id"].duplicated().any(), f"[{name}] train and val share ids"
    df.to_parquet(full_path + ".tmp")
    os.replace(full_path + ".tmp", full_path)
    print(f"[{name}] {len(df)} rows -> {full_path}")
