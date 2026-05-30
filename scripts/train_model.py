"""
Train XGBoost model for NEET college prediction.
Input:  data/training/{train,val,test}_split.csv + label_encoders.pkl
Output: models/neet_xgboost/ (model + metadata)
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    top_k_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import xgboost as xgb


MODEL_DIR = Path("models/neet_xgboost")
TRAINING_DIR = Path("data/training")


def load_data():
    train = pd.read_csv(TRAINING_DIR / "train_split.csv")
    val = pd.read_csv(TRAINING_DIR / "val_split.csv")
    test = pd.read_csv(TRAINING_DIR / "test_split.csv")

    with open(TRAINING_DIR / "metadata.json") as f:
        meta = json.load(f)

    with open(TRAINING_DIR / "label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return train, val, test, meta, encoders


def train(train_df, val_df, num_classes):
    features = [c for c in train_df.columns if c != "college_name_le"]
    X_train = train_df[features].values
    y_train = train_df["college_name_le"].values
    X_val = val_df[features].values
    y_val = val_df["college_name_le"].values

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "learning_rate": 0.15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "seed": 42,
        "verbosity": 1,
    }

    print("Training XGBoost (multi-class college prediction)...")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Classes: {num_classes} colleges")
    print(f"  Features: {features}")
    print()

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=150,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=25,
        verbose_eval=25,
    )

    return model, features

def evaluate(model, df, features, encoders, split_name="test"):
    X = df[features].values
    y = df["college_name_le"].values
    dmat = xgb.DMatrix(X, feature_names=features)

    y_proba = model.predict(dmat)

    # Top-1 accuracy
    y_pred_top1 = y_proba.argmax(axis=1)
    top1_acc = (y_pred_top1 == y).mean()

    # Top-5 accuracy
    top5_acc = top_k_accuracy_score(y, y_proba, k=5, labels=range(y_proba.shape[1]))

    # Top-10 accuracy
    top10_acc = top_k_accuracy_score(y, y_proba, k=min(10, num_classes), labels=range(y_proba.shape[1]))

    print(f"\n{split_name.upper()} Metrics:")
    print(f"  Top-1 Accuracy: {top1_acc:.4f}")
    print(f"  Top-5 Accuracy: {top5_acc:.4f}")
    print(f"  Top-10 Accuracy: {top10_acc:.4f}")

    return {
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "top10_accuracy": float(top10_acc),
    }

def save_artifacts(model, features, metrics, encoders):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_DIR / "model.json"))

    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Inverse map: encoded int → college name
    college_encoder = encoders["college_name"]
    inverse_map = {i: name for i, name in enumerate(college_encoder.classes_)}

    meta = {
        "model_type": "xgboost_multiclass",
        "features": features,
        "target": "college_name_le",
        "num_classes": len(inverse_map),
        "num_boosted_rounds": model.num_boosted_rounds(),
        "metrics": metrics,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Save inverse mapping (needed for inference)
    with open(MODEL_DIR / "college_inverse_map.pkl", "wb") as f:
        pickle.dump(inverse_map, f)

    print(f"\nModel saved to {MODEL_DIR}/")

def feature_importance(model, features):
    scores = model.get_score(importance_type="gain")
    print("\nFeature Importance (gain):")
    total = sum(scores.values()) if scores else 1
    for i, feat in enumerate(features):
        key = f"f{i}"
        imp = scores.get(key, 0)
        pct = imp / total * 100 if total > 0 else 0
        print(f"  {feat:30s} {imp:10.2f} ({pct:.1f}%)")

if __name__ == "__main__":
    train_df, val_df, test_df, meta, encoders = load_data()

    features = meta["features"]
    num_classes = meta["num_colleges"]

    model, features = train(train_df, val_df, num_classes)

    val_metrics = evaluate(model, val_df, features, encoders, "validation")
    test_metrics = evaluate(model, test_df, features, encoders, "test")

    feature_importance(model, features)
    save_artifacts(model, features, test_metrics, encoders)

    print("\nDone!")
