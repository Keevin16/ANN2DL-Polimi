#!/usr/bin/env python3
"""Comprehensive data analysis for the Pirate Pain dataset."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "pirate_pain_train.csv")
LABELS_PATH = os.path.join(DATA_DIR, "pirate_pain_train_labels.csv")

LABELS = ["no_pain", "low_pain", "high_pain"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

def parse_numeric(value):
    """Parse numeric values including word numbers."""
    WORD_TO_INT = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    }
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    value = value.strip().lower()
    if value in WORD_TO_INT:
        return float(WORD_TO_INT[value])
    try:
        return float(value)
    except ValueError:
        return 0.0

def main():
    print("="*80)
    print("PIRATE PAIN DATASET ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n[1] Loading data...")
    X_train = pd.read_csv(TRAIN_PATH, dtype={"sample_index": str})
    y_df = pd.read_csv(LABELS_PATH, dtype={"sample_index": str})
    y_series = y_df.set_index("sample_index")["label"].map(LABEL_TO_IDX)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of unique samples: {X_train['sample_index'].nunique()}")
    print(f"Number of labels: {len(y_series)}")
    
    # Class distribution
    print("\n[2] Class Distribution:")
    class_counts = Counter(y_series.values)
    total = len(y_series)
    for label in LABELS:
        idx = LABEL_TO_IDX[label]
        count = class_counts.get(idx, 0)
        pct = 100.0 * count / total
        print(f"  {label:12s}: {count:4d} ({pct:6.2f}%)")
    
    # Sequence length analysis
    print("\n[3] Sequence Length Analysis:")
    seq_lengths = X_train.groupby("sample_index").size()
    print(f"  Mean length: {seq_lengths.mean():.2f}")
    print(f"  Std length: {seq_lengths.std():.2f}")
    print(f"  Min length: {seq_lengths.min()}")
    print(f"  Max length: {seq_lengths.max()}")
    print(f"  Median length: {seq_lengths.median():.2f}")
    
    # Check if all sequences have same length
    unique_lengths = seq_lengths.unique()
    print(f"  Unique lengths: {sorted(unique_lengths)}")
    
    # Feature analysis
    print("\n[4] Feature Analysis:")
    feature_cols = [c for c in X_train.columns if c.startswith("pain_survey_") or c.startswith("joint_")]
    static_cols = ["n_legs", "n_hands", "n_eyes"]
    print(f"  Temporal features: {len(feature_cols)}")
    print(f"  Static features: {len(static_cols)}")
    
    # Missing values
    print("\n[5] Missing Values:")
    missing = X_train[feature_cols + static_cols].isnull().sum()
    if missing.sum() > 0:
        print("  Missing values found:")
        for col, count in missing[missing > 0].items():
            print(f"    {col}: {count}")
    else:
        print("  No missing values")
    
    # Feature statistics by class
    print("\n[6] Feature Statistics by Class:")
    
    # Parse static features
    for col in static_cols:
        X_train[col + "_parsed"] = X_train[col].apply(parse_numeric)
    
    # Analyze pain_survey features
    print("\n  Pain Survey Features:")
    for col in [c for c in feature_cols if c.startswith("pain_survey_")]:
        print(f"\n    {col}:")
        for label in LABELS:
            idx = LABEL_TO_IDX[label]
            sample_ids = y_series[y_series == idx].index
            values = X_train[X_train["sample_index"].isin(sample_ids)][col]
            print(f"      {label:12s}: mean={values.mean():.3f}, std={values.std():.3f}, "
                  f"min={values.min():.3f}, max={values.max():.3f}")
    
    # Analyze joint features
    joint_cols = [c for c in feature_cols if c.startswith("joint_")]
    print(f"\n  Joint Features (showing first 5 of {len(joint_cols)}):")
    for col in joint_cols[:5]:
        print(f"\n    {col}:")
        for label in LABELS:
            idx = LABEL_TO_IDX[label]
            sample_ids = y_series[y_series == idx].index
            values = X_train[X_train["sample_index"].isin(sample_ids)][col]
            print(f"      {label:12s}: mean={values.mean():.6f}, std={values.std():.6f}")
    
    # Static features by class
    print("\n  Static Features:")
    for col in static_cols:
        print(f"\n    {col}:")
        for label in LABELS:
            idx = LABEL_TO_IDX[label]
            sample_ids = y_series[y_series == idx].index
            values = X_train[X_train["sample_index"].isin(sample_ids)][col + "_parsed"]
            unique_vals = sorted(values.unique())
            print(f"      {label:12s}: unique={unique_vals}, mean={values.mean():.2f}")
    
    # Temporal patterns
    print("\n[7] Temporal Pattern Analysis:")
    # Check if pain_survey values change over time
    for col in [c for c in feature_cols if c.startswith("pain_survey_")]:
        sample_temporal_var = X_train.groupby("sample_index")[col].agg(["mean", "std", "min", "max"])
        print(f"\n  {col} temporal variation:")
        print(f"    Mean std across samples: {sample_temporal_var['std'].mean():.3f}")
        print(f"    Samples with variation: {(sample_temporal_var['std'] > 0.01).sum()} / {len(sample_temporal_var)}")
    
    # Correlation analysis
    print("\n[8] Correlation Analysis:")
    # Pain survey correlations
    pain_survey_cols = [c for c in feature_cols if c.startswith("pain_survey_")]
    pain_corr = X_train[pain_survey_cols].corr()
    print("\n  Pain survey correlations:")
    print(pain_corr)
    
    # Check for highly correlated joint features
    joint_sample = X_train[joint_cols[:10]].corr()  # Sample first 10
    print(f"\n  Joint features (sample) max correlation: {joint_sample.values[np.triu_indices_from(joint_sample.values, k=1)].max():.3f}")
    
    # Class separability
    print("\n[9] Class Separability Analysis:")
    # Aggregate features per sample
    sample_features = []
    sample_labels = []
    
    for sample_id in y_series.index:
        sample_data = X_train[X_train["sample_index"] == sample_id]
        if len(sample_data) == 0:
            continue
        
        # Aggregate temporal features
        agg_features = {}
        for col in pain_survey_cols:
            agg_features[f"{col}_mean"] = sample_data[col].mean()
            agg_features[f"{col}_std"] = sample_data[col].std()
            agg_features[f"{col}_max"] = sample_data[col].max()
            agg_features[f"{col}_min"] = sample_data[col].min()
        
        for col in joint_cols[:10]:  # Sample first 10 joints
            agg_features[f"{col}_mean"] = sample_data[col].mean()
            agg_features[f"{col}_std"] = sample_data[col].std()
        
        for col in static_cols:
            agg_features[col] = parse_numeric(sample_data[col].iloc[0])
        
        sample_features.append(agg_features)
        sample_labels.append(y_series[sample_id])
    
    sample_df = pd.DataFrame(sample_features)
    
    # Check feature importance (simple variance-based)
    print("\n  Feature variance by class:")
    for label in LABELS:
        idx = LABEL_TO_IDX[label]
        label_mask = np.array(sample_labels) == idx
        if label_mask.sum() > 0:
            label_features = sample_df[label_mask]
            print(f"\n    {label}:")
            top_var = label_features.var().sort_values(ascending=False).head(5)
            for feat, var_val in top_var.items():
                print(f"      {feat}: {var_val:.6f}")
    
    # Check for potential issues
    print("\n[10] Potential Issues:")
    
    # Check for constant features
    constant_features = []
    for col in feature_cols:
        if X_train[col].nunique() <= 1:
            constant_features.append(col)
    if constant_features:
        print(f"  Constant features found: {constant_features}")
    else:
        print("  No constant features")
    
    # Check for very sparse features (mostly zeros)
    sparse_features = []
    for col in joint_cols:
        zero_ratio = (X_train[col] == 0).mean()
        if zero_ratio > 0.9:
            sparse_features.append((col, zero_ratio))
    if sparse_features:
        print(f"  Very sparse features (>90% zeros): {len(sparse_features)}")
        print(f"    Examples: {sparse_features[:5]}")
    else:
        print("  No extremely sparse features")
    
    # Check label distribution per sequence length
    print("\n[11] Label Distribution by Sequence Length:")
    seq_lengths_df = seq_lengths.reset_index()
    seq_lengths_df.columns = ["sample_index", "length"]
    seq_lengths_df["label"] = seq_lengths_df["sample_index"].map(y_series)
    
    for length in sorted(seq_lengths_df["length"].unique()):
        length_data = seq_lengths_df[seq_lengths_df["length"] == length]
        if len(length_data) > 0:
            label_dist = length_data["label"].value_counts().sort_index()
            print(f"\n  Length {length}: {len(length_data)} samples")
            for idx, count in label_dist.items():
                print(f"    {LABELS[idx]}: {count}")

if __name__ == "__main__":
    main()

