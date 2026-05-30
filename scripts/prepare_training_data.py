"""
Prepare training dataset for college prediction.
Given: rank, category, quota, state, year → predict which college.

Input:  data/cleaned/normalized_dataset.csv
Output: data/training/{train,val,test}_split.csv + metadata.json + label_encoders.json
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


MIN_COLLEGE_ROWS = 3
MIN_YEAR = 2020


def prepare():
    src = Path("data/cleaned/merged_dataset_v2.csv")
    out = Path("data/training")
    out.mkdir(exist_ok=True)

    df = pd.read_csv(src)
    print(f"Original rows: {len(df)}")

    # Filter: both ranks present + year >= 2020 + no junk + valid college name
    clean = df[
        (df["opening_rank"].notna())
        & (df["closing_rank"].notna())
        & (df["year"] >= MIN_YEAR)
        & (~df["college_name"].str.contains(r"payment|notice|facility|www\.", case=False, na=False))
        & (df["college_name"] != "Unknown")
        & (~df["college_name"].str.startswith("("))
        & (~df["college_name"].str.match(r"^\d"))
    ].copy()
    print(f"After filtering: {len(clean)}")

    # Cap ranks
    clean["opening_rank"] = clean["opening_rank"].clip(1, 2_000_000).astype(int)
    clean["closing_rank"] = clean["closing_rank"].clip(1, 2_000_000).astype(int)
    clean["year"] = clean["year"].astype(int)
    if "round_number" in clean.columns:
        clean["round_number"] = clean["round_number"].fillna(0).astype(int)
    elif "round" in clean.columns:
        clean["round_number"] = clean["round"].fillna(0).astype(int)
    else:
        clean["round_number"] = 0

    # Compute derived features
    clean["seat_ratio"] = clean.get("seat_count", pd.Series(0)).fillna(0) / 100.0
    clean["historical_cutoff_gap"] = (clean["closing_rank"] - clean["opening_rank"]).clip(0, 2_000_000)

    # Filter colleges with enough data
    college_counts = clean["college_name"].value_counts()
    valid_colleges = college_counts[college_counts >= MIN_COLLEGE_ROWS].index
    clean = clean[clean["college_name"].isin(valid_colleges)].copy()
    print(f"After college filter (min {MIN_COLLEGE_ROWS} rows): {len(clean)} rows, {clean['college_name'].nunique()} colleges")

    # ---- Encode categorical labels (save encoders for inference) ----
    label_encoders = {}

    for col in ["category", "quota", "state", "college_name"]:
        le = LabelEncoder()
        clean[f"{col}_le"] = le.fit_transform(clean[col].astype(str))
        label_encoders[col] = {cls: int(i) for i, cls in enumerate(le.classes_)}
        print(f"  {col}: {len(le.classes_)} classes")

    # Save label encoders (needed for inference)
    with open(out / "label_encoders.json", "w") as f:
        json.dump(label_encoders, f, indent=2)
    with open(out / "label_encoders.pkl", "wb") as f:
        pickle.dump(
            {col: LabelEncoder().fit(clean[col].astype(str)) for col in ["category", "quota", "state", "college_name"]},
            f,
        )

    # ---- Feature columns ----
    FEATURES = [
        "opening_rank",
        "closing_rank",
        "year",
        "round_number",
        "category_le",
        "quota_le",
        "state_le",
        "seat_ratio",
        "historical_cutoff_gap",
    ]
    TARGET = "college_name_le"

    # Stratified split on college_name (use stratify=None for val/test to avoid single-member class error)
    train, temp = train_test_split(clean, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Save splits (only features + target)
    for name, split in [("train_split", train), ("val_split", val), ("test_split", test)]:
        split[FEATURES + [TARGET]].to_csv(out / f"{name}.csv", index=False)

    print(f"\nSaved:")
    print(f"  train_split.csv: {len(train)} rows")
    print(f"  val_split.csv:   {len(val)} rows")
    print(f"  test_split.csv:  {len(test)} rows")

    meta = {
        "features": FEATURES,
        "target": TARGET,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "total_rows": len(clean),
        "num_colleges": int(clean["college_name"].nunique()),
        "min_college_rows": MIN_COLLEGE_ROWS,
        "label_encoders_path": str(out / "label_encoders.pkl"),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  metadata.json + label_encoders.json saved")
    print(f"\nFeatures: {FEATURES}")
    print(f"Target: {TARGET} ({meta['num_colleges']} colleges)")


if __name__ == "__main__":
    prepare()
