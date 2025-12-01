#!/usr/bin/env python3
import os
import math
import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "pirate_pain_train.csv")
LABELS_PATH = os.path.join(DATA_DIR, "pirate_pain_train_labels.csv")
TEST_PATH = os.path.join(DATA_DIR, "pirate_pain_test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Submissions")
os.makedirs(OUT_DIR, exist_ok=True)


def percentile(values: np.ndarray, q: float) -> float:
	"""Robust percentile helper handling NaNs."""
	if values.size == 0:
		return np.nan
	return float(np.nanpercentile(values, q))


def autocorr(x: np.ndarray, lag: int) -> float:
	"""Autocorrelation at given lag, NaN-safe."""
	if len(x) <= lag or lag <= 0:
		return np.nan
	x0 = x[:-lag]
	x1 = x[lag:]
	x0 = x0 - np.nanmean(x0)
	x1 = x1 - np.nanmean(x1)
	den = np.nanstd(x0) * np.nanstd(x1)
	if den == 0 or np.isnan(den):
		return np.nan
	return float(np.nansum(x0 * x1) / ((len(x0)) * den))


def fft_topk_energy(x: np.ndarray, k: int = 3) -> Tuple[float, float, float]:
	"""Top-k FFT magnitude features (excluding DC). Returns fixed 3-tuple padded with NaNs."""
	if len(x) == 0:
		return (np.nan, np.nan, np.nan)
	# zero-mean for stability
	x = x - np.nanmean(x)
	spec = np.fft.rfft(np.nan_to_num(x, nan=0.0))
	mag = np.abs(spec)
	if mag.size <= 1:
		return (np.nan, np.nan, np.nan)
	# exclude DC (index 0)
	mag = mag[1:]
	idx = np.argsort(mag)[::-1]
	top = mag[idx[:k]]
	out = list(top.astype(float))
	while len(out) < 3:
		out.append(np.nan)
	return tuple(out[:3])


def summarize_series(df: pd.DataFrame, sample_id: str) -> Dict[str, float]:
	"""Aggregate per-sample time-series into fixed-length feature vector."""
	sub = df[df["sample_index"] == sample_id]
	sub = sub.sort_values("time")

	# Identify column groups
	survey_cols = [c for c in sub.columns if c.startswith("pain_survey_")]
	joint_cols = [c for c in sub.columns if c.startswith("joint_")]
	static_cols = ["n_legs", "n_hands", "n_eyes"]

	features: Dict[str, float] = {}

	# Static categorical features (take the first occurrence)
	for c in static_cols:
		val = sub[c].iloc[0] if len(sub) > 0 else np.nan
		features[f"static__{c}"] = val

	# Survey features: treat as categorical-ish small integers, collect stats
	for c in survey_cols:
		x = sub[c].values.astype(float)
		if len(x) == 0:
			features[f"{c}__mean"] = np.nan
			features[f"{c}__std"] = np.nan
			features[f"{c}__min"] = np.nan
			features[f"{c}__max"] = np.nan
			features[f"{c}__p25"] = np.nan
			features[f"{c}__p75"] = np.nan
			features[f"{c}__first"] = np.nan
			features[f"{c}__last"] = np.nan
			features[f"{c}__delta"] = np.nan
			for lag in (1, 2, 4, 8):
				features[f"{c}__autocorr_lag{lag}"] = np.nan
			continue
		features[f"{c}__mean"] = float(np.nanmean(x))
		features[f"{c}__std"] = float(np.nanstd(x))
		features[f"{c}__min"] = float(np.nanmin(x)) if len(x) else np.nan
		features[f"{c}__max"] = float(np.nanmax(x)) if len(x) else np.nan
		features[f"{c}__p25"] = percentile(x, 25)
		features[f"{c}__p75"] = percentile(x, 75)
		features[f"{c}__first"] = float(x[0]) if len(x) else np.nan
		features[f"{c}__last"] = float(x[-1]) if len(x) else np.nan
		features[f"{c}__delta"] = float(x[-1] - x[0]) if len(x) else np.nan
		for lag in (1, 2, 4, 8):
			features[f"{c}__autocorr_lag{lag}"] = autocorr(x, lag)

	# Joint features: richer statistics
	for c in joint_cols:
		x = sub[c].values.astype(float)
		# basic stats
		if len(x) == 0:
			features[f"{c}__mean"] = np.nan
			features[f"{c}__std"] = np.nan
			features[f"{c}__min"] = np.nan
			features[f"{c}__max"] = np.nan
			features[f"{c}__p10"] = np.nan
			features[f"{c}__p50"] = np.nan
			features[f"{c}__p90"] = np.nan
			features[f"{c}__diff_mean"] = np.nan
			features[f"{c}__diff_std"] = np.nan
			features[f"{c}__diff_abs_mean"] = np.nan
			features[f"{c}__energy"] = np.nan
			features[f"{c}__fft_top1"] = np.nan
			features[f"{c}__fft_top2"] = np.nan
			features[f"{c}__fft_top3"] = np.nan
			for lag in (1, 2, 4, 8, 16):
				features[f"{c}__autocorr_lag{lag}"] = np.nan
			features[f"{c}__first"] = np.nan
			features[f"{c}__last"] = np.nan
			features[f"{c}__delta"] = np.nan
			continue
		features[f"{c}__mean"] = float(np.nanmean(x))
		features[f"{c}__std"] = float(np.nanstd(x))
		features[f"{c}__min"] = float(np.nanmin(x))
		features[f"{c}__max"] = float(np.nanmax(x))
		features[f"{c}__p10"] = percentile(x, 10)
		features[f"{c}__p50"] = percentile(x, 50)
		features[f"{c}__p90"] = percentile(x, 90)
		# temporal deltas
		if len(x) >= 2:
			dx = np.diff(x)
			features[f"{c}__diff_mean"] = float(np.nanmean(dx))
			features[f"{c}__diff_std"] = float(np.nanstd(dx))
			features[f"{c}__diff_abs_mean"] = float(np.nanmean(np.abs(dx)))
		else:
			features[f"{c}__diff_mean"] = np.nan
			features[f"{c}__diff_std"] = np.nan
			features[f"{c}__diff_abs_mean"] = np.nan
		# energy and fft
		features[f"{c}__energy"] = float(np.nansum(x * x))
		f1, f2, f3 = fft_topk_energy(x, k=3)
		features[f"{c}__fft_top1"] = f1
		features[f"{c}__fft_top2"] = f2
		features[f"{c}__fft_top3"] = f3
		# autocorrelations
		for lag in (1, 2, 4, 8, 16):
			features[f"{c}__autocorr_lag{lag}"] = autocorr(x, lag)
		# endpoints
		features[f"{c}__first"] = float(x[0]) if len(x) else np.nan
		features[f"{c}__last"] = float(x[-1]) if len(x) else np.nan
		features[f"{c}__delta"] = float(x[-1] - x[0]) if len(x) else np.nan

	# Global sequence metadata
	features["seq__length"] = int(len(sub))

	return features


def build_feature_table(train_df: pd.DataFrame, ids: List[str]) -> pd.DataFrame:
	records = []
	for sid in ids:
		records.append(summarize_series(train_df, sid))
	return pd.DataFrame.from_records(records, index=ids)


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str], List[str]]:
	X_train = pd.read_csv(TRAIN_PATH, dtype={"sample_index": str})
	y_df = pd.read_csv(LABELS_PATH, dtype={"sample_index": str})
	X_test = pd.read_csv(TEST_PATH, dtype={"sample_index": str})
	# IDs
	train_ids = y_df["sample_index"].astype(str).tolist()
	test_ids = sorted(list(X_test["sample_index"].astype(str).unique()))
	return X_train, y_df.set_index("sample_index")["label"], X_test, train_ids, test_ids


def main():
	print("Loading data...")
	train_ts, y_series, test_ts, train_ids, test_ids = load_data()

	print("Building features (train)...")
	X_train_feat = build_feature_table(train_ts, train_ids)
	X_train_feat.index.name = "sample_index"
	y = y_series.loc[X_train_feat.index]

	print("Building features (test)...")
	X_test_feat = build_feature_table(test_ts, test_ids)
	X_test_feat.index.name = "sample_index"

	# Identify categorical and numeric columns
	cat_cols = [c for c in X_train_feat.columns if c.startswith("static__")]
	num_cols = [c for c in X_train_feat.columns if c not in cat_cols]

	num_pipe = Pipeline([
		("impute", SimpleImputer(strategy="median")),
		("scale", StandardScaler(with_mean=True, with_std=True)),
	])
	cat_pipe = Pipeline([
		("impute", SimpleImputer(strategy="most_frequent")),
		("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
	])
	preprocess = ColumnTransformer([
		("num", num_pipe, num_cols),
		("cat", cat_pipe, cat_cols),
	], remainder="drop")

	# Two complementary models to ensemble: HGB and LogisticRegression with balanced classes
	hgb = HistGradientBoostingClassifier(
		max_depth=8,
		learning_rate=0.08,
		max_bins=255,
		l2_regularization=1e-3,
		min_samples_leaf=20,
		early_stopping=True,
		random_state=42,
	)
	logreg = LogisticRegression(
		C=1.0,
		max_iter=2000,
		class_weight="balanced",
		n_jobs=None,
		multi_class="auto",
	)

	pipeline_hgb = Pipeline([("prep", preprocess), ("clf", hgb)])
	pipeline_lr = Pipeline([("prep", preprocess), ("clf", logreg)])

	# Cross-validation with stratification on labels (one sample per subject)
	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	oof_preds_hgb = np.zeros((len(X_train_feat), 3), dtype=float)
	oof_preds_lr = np.zeros((len(X_train_feat), 3), dtype=float)
	labels = ["no_pain", "low_pain", "high_pain"]
	label_to_idx = {l: i for i, l in enumerate(labels)}
	y_idx = y.map(label_to_idx).values

	print("Starting 5-fold CV...")
	for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_feat, y)):
		X_tr, X_va = X_train_feat.iloc[tr_idx], X_train_feat.iloc[va_idx]
		y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

		print(f"Fold {fold+1}/5: train={len(X_tr)}, valid={len(X_va)}")
		# Fit HGB
		pipeline_hgb.fit(X_tr, y_tr)
		proba_hgb = pipeline_hgb.predict_proba(X_va)
		oof_preds_hgb[va_idx] = proba_hgb
		# Fit LR
		pipeline_lr.fit(X_tr, y_tr)
		proba_lr = pipeline_lr.predict_proba(X_va)
		oof_preds_lr[va_idx] = proba_lr

	# Ensemble OOF predictions (simple average)
	oof_proba = 0.6 * oof_preds_hgb + 0.4 * oof_preds_lr
	oof_pred_idx = np.argmax(oof_proba, axis=1)
	oof_pred = [labels[i] for i in oof_pred_idx]
	macro_f1 = f1_score(y, oof_pred, average="macro")
	print(f"OOF Macro-F1: {macro_f1:.5f}")
	print("Per-class report:")
	print(classification_report(y, oof_pred, digits=4))

	# Fit on full data for submission
	print("Fitting full models for test prediction...")
	pipeline_hgb.fit(X_train_feat, y)
	pipeline_lr.fit(X_train_feat, y)

	full_df = pd.concat([X_train_feat, X_test_feat], axis=0)
	proba_hgb_test = pipeline_hgb.predict_proba(full_df)[-len(X_test_feat):]
	proba_lr_test = pipeline_lr.predict_proba(full_df)[-len(X_test_feat):]
	test_proba = 0.6 * proba_hgb_test + 0.4 * proba_lr_test
	test_pred_idx = np.argmax(test_proba, axis=1)
	test_pred = [labels[i] for i in test_pred_idx]

	# Save metrics and submission
	metrics_path = os.path.join(OUT_DIR, "tabular_baseline_metrics.json")
	with open(metrics_path, "w") as f:
		json.dump({"oof_macro_f1": macro_f1}, f, indent=2)
	print(f"Saved metrics to: {metrics_path}")

	sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
	sub_df = pd.DataFrame({"sample_index": sample_sub["sample_index"], "label": test_pred})
	sub_path = os.path.join(OUT_DIR, "tabular_baseline_submission.csv")
	sub_df.to_csv(sub_path, index=False)
	print(f"Saved submission to: {sub_path}")


if __name__ == "__main__":
	main()


