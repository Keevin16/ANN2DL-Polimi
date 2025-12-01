#!/usr/bin/env python3
import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_DIR = "/kaggle/input/the-pirate-pain-dataset"
TRAIN_PATH = os.path.join(DATA_DIR, "pirate_pain_train.csv")
LABELS_PATH = os.path.join(DATA_DIR, "pirate_pain_train_labels.csv")
TEST_PATH = os.path.join(DATA_DIR, "pirate_pain_test.csv")
SUBMISSION_SAMPLE = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_DIR = "/kaggle/output"
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = ["no_pain", "low_pain", "high_pain"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}


def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
	if torch.backends.mps.is_available():
		return torch.device("mps")
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


WORD_TO_INT = {
	"zero": 0,
	"one": 1,
	"two": 2,
	"three": 3,
	"four": 4,
	"five": 5,
	"six": 6,
	"seven": 7,
	"eight": 8,
	"nine": 9,
}


def parse_numeric(value: str) -> float:
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


def load_frames() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
	X_train = pd.read_csv(TRAIN_PATH, dtype={"sample_index": str})
	y_df = pd.read_csv(LABELS_PATH, dtype={"sample_index": str})
	X_test = pd.read_csv(TEST_PATH, dtype={"sample_index": str})
	y_series = y_df.set_index("sample_index")["label"].map(LABEL_TO_IDX)
	return X_train, y_series, X_test


def create_raw_sequences(df: pd.DataFrame, sample_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
	# Get feature columns, excluding constant features
	all_joint_cols = [c for c in df.columns if c.startswith("joint_")]
	all_pain_survey_cols = [c for c in df.columns if c.startswith("pain_survey_")]
	
	# Remove constant features (joint_30 is constant)
	joint_cols = [c for c in all_joint_cols if c != "joint_30"]
	pain_survey_cols = all_pain_survey_cols
	
	feature_cols = pain_survey_cols + joint_cols
	static_cols = ["n_legs", "n_hands", "n_eyes"]
	time_steps = 160  # validated earlier; all sequences have equal length
	# Static features: 3 original + 1 all_normal indicator = 4 total
	static_dim_extended = len(static_cols) + 1
	feature_dim = len(feature_cols) + static_dim_extended

	raw_sequences = np.zeros((len(sample_ids), time_steps, feature_dim), dtype=np.float32)
	static_per_sample = np.zeros((len(sample_ids), static_dim_extended), dtype=np.float32)

	for idx, sid in enumerate(sample_ids):
		sub = df[df["sample_index"] == sid].sort_values("time")
		
		# Extract temporal features - use pain_survey as-is (model will learn the relationship)
		# Analysis shows high_pain has LOWER pain_survey values, which is discriminative
		values = sub[feature_cols].to_numpy(dtype=np.float32)
		
		if values.shape[0] != time_steps:
			# simple padding/truncation fallback
			temp = np.zeros((time_steps, len(feature_cols)), dtype=np.float32)
			length = min(time_steps, values.shape[0])
			temp[:length] = values[:length]
			values = temp

		# Enhanced static features: add "all_normal" indicator
		# Analysis shows ALL high_pain samples have n_legs=2, n_hands=2, n_eyes=2
		static_vals = []
		for sc in static_cols:
			raw_val = sub[sc].iloc[0] if not sub.empty else 0
			static_vals.append(parse_numeric(raw_val))
		
		# Add all_normal indicator: 1.0 if all are 2, else 0.0
		all_normal = 1.0 if all(v == 2.0 for v in static_vals) else 0.0
		static_vals.append(all_normal)
		
		# Store extended static features (original 3 + all_normal = 4)
		static_per_sample[idx] = np.array(static_vals, dtype=np.float32)
		static_arr = np.tile(np.array(static_vals, dtype=np.float32), (time_steps, 1))

		raw_sequences[idx] = np.concatenate([values, static_arr], axis=1)

	# Return feature names (static_cols + "all_normal" for the extended static features)
	extended_static_cols = static_cols + ["all_normal"]
	return raw_sequences, static_per_sample, feature_cols + extended_static_cols


def normalize(train_seq: np.ndarray, val_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	train_data = train_seq[..., :-1]
	train_mask = train_seq[..., -1:]
	mask_sum = np.clip(train_mask.sum(axis=(0, 1)), 1.0, None)
	mean = (train_data * train_mask).sum(axis=(0, 1)) / mask_sum
	var = ((train_data - mean) ** 2 * train_mask).sum(axis=(0, 1)) / mask_sum
	std = np.sqrt(var)
	std[std < 1e-6] = 1.0

	def apply_norm(arr: np.ndarray) -> np.ndarray:
		data = arr[..., :-1]
		mask = arr[..., -1:]
		data = ((data - mean) / std) * mask
		return np.concatenate([data, mask], axis=-1)

	return apply_norm(train_seq), apply_norm(val_seq), mean, std


class SequenceDataset(Dataset):
	def __init__(
		self,
		sequences: np.ndarray,
		static_feats: np.ndarray,
		labels: np.ndarray,
		is_train: bool,
		jitter_std: float = 0.02,
		time_mask_prob: float = 0.25,
		time_mask_max: int = 10,
		time_shift_max: int = 4,
		adaptive_augment: bool = True,
	):
		self.X = torch.from_numpy(sequences.astype(np.float32))
		self.S = torch.from_numpy(static_feats.astype(np.float32))
		self.y = torch.from_numpy(labels.astype(np.int64))
		self.is_train = is_train
		self.jitter_std = jitter_std
		self.time_mask_prob = time_mask_prob
		self.time_mask_max = time_mask_max
		self.time_shift_max = time_shift_max
		self.adaptive_augment = adaptive_augment
		
		# Compute class-specific augmentation multipliers
		if adaptive_augment and is_train:
			# Explicit augmentation multipliers: heavy for high_pain, moderate for low_pain
			# no_pain=0, low_pain=1, high_pain=2
			self.augment_multipliers = np.array([1.0, 2.5, 4.0], dtype=np.float32)
			# high_pain gets 4x augmentation, low_pain gets 2.5x, no_pain gets 1x
		else:
			self.augment_multipliers = np.ones(len(LABELS), dtype=np.float32)

	def __len__(self):
		return self.X.shape[0]

	def _augment(self, seq: torch.Tensor, label: int) -> torch.Tensor:
		data = seq[:, :-1]
		mask = seq[:, -1:]
		valid_len = int(mask.sum().item())
		
		# Adaptive augmentation: apply stronger augmentation to minority classes
		# high_pain gets 4x augmentation, low_pain gets 2.5x, no_pain gets 1x
		if self.adaptive_augment and self.is_train:
			multiplier = self.augment_multipliers[label]
		else:
			multiplier = 1.0

		if self.jitter_std > 0 and valid_len > 0:
			# Increase jitter for minority classes (high_pain: 4x, low_pain: 2.5x)
			effective_jitter = self.jitter_std * multiplier
			noise = torch.randn_like(data) * effective_jitter
			data = data + noise * mask

		if self.time_shift_max > 0 and valid_len > 0:
			# Increase shift range for minority classes
			effective_shift_max = int(self.time_shift_max * multiplier)
			shift = random.randint(-effective_shift_max, effective_shift_max)
			if shift != 0:
				if shift > 0:
					data = torch.cat([torch.zeros_like(data[:shift]), data[:-shift]], dim=0)
					mask = torch.cat([torch.zeros_like(mask[:shift]), mask[:-shift]], dim=0)
				else:
					shift = abs(shift)
					data = torch.cat([data[shift:], torch.zeros_like(data[:shift])], dim=0)
					mask = torch.cat([mask[shift:], torch.zeros_like(mask[:shift])], dim=0)

		if self.time_mask_prob > 0 and valid_len > 0:
			# Increase mask probability for minority classes
			effective_mask_prob = min(1.0, self.time_mask_prob * multiplier)
			if random.random() < effective_mask_prob:
				effective_mask_max = int(self.time_mask_max * multiplier)
				mask_len = random.randint(1, min(effective_mask_max, valid_len))
				start = random.randint(0, max(valid_len - mask_len, 0))
				data[start:start + mask_len] = 0.0

		return torch.cat([data, mask], dim=1)

	def __getitem__(self, idx: int):
		x = self.X[idx].clone()
		label = self.y[idx].item()
		if self.is_train:
			x = self._augment(x, label)
		return x, self.S[idx], self.y[idx]


@dataclass
class TrainConfig:
	input_dim: int
	static_dim: int = 4  # Increased: 3 original + 1 all_normal indicator
	hidden_dim: int = 192
	num_layers: int = 2
	dropout: float = 0.4  # Increased for better regularization
	bidirectional: bool = True
	max_epochs: int = 50
	batch_size: int = 64
	lr: float = 3e-4  # Lower LR for more stable training
	weight_decay: float = 1e-3  # Increased weight decay
	patience: int = 8  # Reduced: stop earlier if F1 plateaus (was 12)
	patience_loss: int = 8  # Early stop on val loss if no improvement
	lr_patience: int = 5  # Reduce LR if F1 doesn't improve for this many epochs
	lr_factor: float = 0.5  # Factor to reduce LR by
	label_smoothing: float = 0.05
	jitter_std: float = 0.02
	time_mask_prob: float = 0.3
	time_mask_max: int = 12
	time_shift_max: int = 6
	window_sizes: Tuple[int, ...] = (96,)  # Balance: captures 60% context, gives ~5 windows per sequence
	stride: int = 16  # Overlap for data augmentation (~3,305 total windows vs 661 sequences)
	warmup_epochs: int = 3
	use_attention: bool = True  # Add attention mechanism
	use_focal_loss: bool = False  # Use standard CrossEntropyLoss - simpler and more stable
	focal_gamma: float = 2.0  # Not used when use_focal_loss=False


class Attention(nn.Module):
	"""Simple attention mechanism for temporal features."""
	def __init__(self, hidden_dim: int):
		super().__init__()
		self.attention = nn.Linear(hidden_dim, 1)
	
	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		# hidden_states: [batch, seq_len, hidden_dim]
		attn_weights = self.attention(hidden_states)  # [batch, seq_len, 1]
		attn_weights = torch.softmax(attn_weights, dim=1)
		context = torch.sum(attn_weights * hidden_states, dim=1)  # [batch, hidden_dim]
		return context


class GRUClassifier(nn.Module):
	def __init__(self, config: TrainConfig, num_classes: int):
		super().__init__()
		self.config = config
		self.gru = nn.GRU(
			input_size=config.input_dim,
			hidden_size=config.hidden_dim,
			num_layers=config.num_layers,
			batch_first=True,
			dropout=config.dropout if config.num_layers > 1 else 0.0,
			bidirectional=config.bidirectional,
		)
		direction_factor = 2 if config.bidirectional else 1
		self.dropout = nn.Dropout(config.dropout)
		
		# Attention mechanism for better temporal modeling
		if config.use_attention:
			self.attention = Attention(config.hidden_dim * direction_factor)
		else:
			self.attention = None
		
		# Static feature pathway - enhanced to leverage all_normal feature
		# all_normal (index 3) is highly discriminative: all high_pain have it=1
		self.static_mlp = nn.Sequential(
			nn.Linear(config.static_dim, 128),  # Increased capacity
			nn.ReLU(),
			nn.Dropout(config.dropout * 0.5),
			nn.Linear(128, 64),  # Deeper network
			nn.ReLU(),
			nn.Dropout(config.dropout * 0.5),
			nn.Linear(64, 32),
		)
		
		# Final classifier - combine temporal and static features
		gru_output_dim = config.hidden_dim * direction_factor
		fc_input_dim = gru_output_dim + 32
		self.fc = nn.Sequential(
			nn.Linear(fc_input_dim, 128),
			nn.ReLU(),
			nn.Dropout(config.dropout * 0.5),
			nn.Linear(128, num_classes),
		)

	def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
		out, h = self.gru(x)  # out: [batch, seq_len, hidden_dim * directions]
		
		if self.attention is not None:
			# Use attention over all timesteps
			gru_feat = self.attention(out)
		else:
			# Use last hidden state (bidirectional: concat both directions)
			if self.config.bidirectional:
				gru_feat = torch.cat([h[-2], h[-1]], dim=1)
			else:
				gru_feat = h[-1]
		
		gru_feat = self.dropout(gru_feat)
		s_emb = self.static_mlp(s)
		feat = torch.cat([gru_feat, s_emb], dim=1)
		return self.fc(feat)


def compute_class_weights(labels: np.ndarray, method: str = "inverse_freq") -> torch.Tensor:
	"""Compute class weights using inverse frequency (more aggressive for imbalanced classes)."""
	class_counts = np.bincount(labels, minlength=len(LABELS))
	class_counts[class_counts == 0] = 1
	
	if method == "inverse_freq":
		# Inverse frequency: weight = total / count
		total = len(labels)
		weights = total / class_counts.astype(np.float32)
	elif method == "sqrt_inverse":
		# Square root of inverse frequency (less aggressive)
		total = len(labels)
		weights = np.sqrt(total / class_counts.astype(np.float32))
	else:
		# Balanced: weight = total / (num_classes * count)
		total = len(labels)
		weights = total / (len(LABELS) * class_counts.astype(np.float32))
	
	# Normalize so smallest weight is 1.0
	weights = weights / weights.min()
	return torch.from_numpy(weights.astype(np.float32))


class FocalLoss(nn.Module):
	"""Focal loss for handling class imbalance."""
	def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
		super().__init__()
		if alpha is not None:
			self.register_buffer("alpha", alpha)
		else:
			self.alpha = None
		self.gamma = gamma
		self.reduction = reduction
	
	def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
		pt = torch.exp(-ce_loss)
		focal_loss = ((1 - pt) ** self.gamma) * ce_loss
		
		if self.reduction == "mean":
			return focal_loss.mean()
		elif self.reduction == "sum":
			return focal_loss.sum()
		else:
			return focal_loss


def find_optimal_thresholds(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
	"""
	Find optimal per-class thresholds to maximize macro F1.
	Uses a simplified grid search focusing on minority classes.
	"""
	from sklearn.metrics import f1_score
	
	best_thresholds = np.ones(len(LABELS))  # Default: no threshold adjustment
	best_f1 = f1_score(labels, probs.argmax(axis=1), average="macro")
	
	# Focus search on minority classes (low_pain=1, high_pain=2)
	# Keep no_pain threshold at 1.0, search for others
	thresholds_to_try = np.linspace(0.7, 1.8, 12)  # Reduced search space
	
	for thresh_1 in thresholds_to_try:
		for thresh_2 in thresholds_to_try:
			thresholds = np.array([1.0, thresh_1, thresh_2])  # Keep no_pain at 1.0
			adjusted_probs = probs * thresholds
			adjusted_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True)  # Renormalize
			preds = adjusted_probs.argmax(axis=1)
			f1 = f1_score(labels, preds, average="macro")
			
			if f1 > best_f1 + 1e-6:  # Small epsilon to avoid numerical issues
				best_f1 = f1
				best_thresholds = thresholds
	
	return best_thresholds


def evaluate(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	window_owners: Optional[np.ndarray] = None,
	num_subjects: Optional[int] = None,
	class_thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
	"""
	Evaluate model. If window_owners is provided, aggregate to subject level before computing F1.
	class_thresholds: Optional per-class probability multipliers for threshold tuning.
	"""
	model.eval()
	all_logits = []
	all_labels = []
	criterion = nn.CrossEntropyLoss()
	total_loss = 0.0
	with torch.no_grad():
		for xb, sb, yb in loader:
			xb = xb.to(device)
			sb = sb.to(device)
			yb = yb.to(device)
			logits = model(xb, sb)
			loss = criterion(logits, yb)
			total_loss += loss.item() * xb.size(0)
			all_logits.append(logits.cpu().numpy())
			all_labels.append(yb.cpu().numpy())
	all_logits = np.concatenate(all_logits, axis=0)
	all_labels = np.concatenate(all_labels, axis=0)
	probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
	
	# Apply class thresholds if provided (for threshold tuning)
	if class_thresholds is not None:
		probs = probs * class_thresholds
		probs = probs / probs.sum(axis=1, keepdims=True)  # Renormalize
	
	# Aggregate to subject level if window owners provided
	if window_owners is not None and num_subjects is not None:
		subject_probs = aggregate_subject_probs(probs, window_owners.tolist(), num_subjects)
		# Get subject labels: each window has the label of its owner subject
		# Since all windows from the same subject have the same label, we can use the first window for each subject
		subject_labels = np.zeros(num_subjects, dtype=np.int64)
		for subj_idx in range(num_subjects):
			win_mask = window_owners == subj_idx
			if win_mask.any():
				# All windows from this subject have the same label, so take first
				subject_labels[subj_idx] = all_labels[win_mask][0]
		subject_preds = subject_probs.argmax(axis=1)
		f1 = f1_score(subject_labels, subject_preds, average="macro")
		return f1, subject_probs, subject_labels
	else:
		# Window-level evaluation
		preds = probs.argmax(axis=1)
		f1 = f1_score(all_labels, preds, average="macro")
		return f1, probs, all_labels


def train_one_fold(
	fold: int,
	config: TrainConfig,
	train_seq: np.ndarray,
	train_static: np.ndarray,
	train_labels: np.ndarray,
	val_seq: np.ndarray,
	val_static: np.ndarray,
	val_labels: np.ndarray,
	val_window_owners: np.ndarray,
	num_val_subjects: int,
	device: torch.device,
) -> Tuple[np.ndarray, Dict]:
	set_seed(42 + fold)

	train_seq_norm, val_seq_norm, mean, std = normalize(train_seq, val_seq)

	train_ds = SequenceDataset(
		train_seq_norm,
		train_static,
		train_labels,
		is_train=True,
		jitter_std=config.jitter_std,
		time_mask_prob=config.time_mask_prob,
		time_mask_max=config.time_mask_max,
		time_shift_max=config.time_shift_max,
	)
	val_ds = SequenceDataset(
		val_seq_norm,
		val_static,
		val_labels,
		is_train=False,
		jitter_std=0.0,
		time_mask_prob=0.0,
		time_mask_max=0,
		time_shift_max=0,
	)

	# Create weighted sampler for oversampling minority classes
	# More aggressive oversampling: balance to majority class size
	class_counts = np.bincount(train_labels, minlength=len(LABELS))
	max_count = class_counts.max()
	
	# Calculate target samples per class (balance to majority)
	target_counts = np.full(len(LABELS), max_count, dtype=np.int32)
	
	# Compute sample weights: minority classes get much higher weights
	class_weights_sampler = target_counts.astype(np.float32) / (class_counts.astype(np.float32) + 1e-8)
	sample_weights = torch.from_numpy(class_weights_sampler[train_labels].astype(np.float32))
	
	# Use more aggressive oversampling: sample up to 2x the dataset size
	num_samples = int(len(train_ds) * 1.5)  # 50% more samples
	sampler = torch.utils.data.WeightedRandomSampler(
		weights=sample_weights,
		num_samples=num_samples,
		replacement=True,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=config.batch_size,
		sampler=sampler,
		drop_last=False,
	)
	val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

	model = GRUClassifier(config, num_classes=len(LABELS)).to(device)

	# Setup loss function with simple inverse frequency weighting (no extra multipliers)
	class_weights = compute_class_weights(train_labels, method="inverse_freq").to(device)
	# No extra multipliers - let inverse frequency do the work
	
	if config.use_focal_loss:
		criterion = FocalLoss(alpha=class_weights, gamma=config.focal_gamma).to(device)
	else:
		criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	# Use ReduceLROnPlateau for adaptive LR reduction based on F1 score
	# This helps when F1 plateaus - reduces LR to fine-tune
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode='max',  # Maximize F1 score
		factor=config.lr_factor,
		patience=config.lr_patience,
		verbose=True,
		min_lr=config.lr * 0.01,  # Don't go below 1% of initial LR
	)
	
	if config.warmup_epochs > 0:
		initial_lr = config.lr / max(1, config.warmup_epochs)
		for param_group in optimizer.param_groups:
			param_group["lr"] = initial_lr

	best_state = None
	best_f1 = -1.0
	best_val_loss = float("inf")
	epochs_no_improve_f1 = 0
	epochs_no_improve_loss = 0

	# Create a non-oversampled loader for training loss computation (reused each epoch)
	train_loader_eval = DataLoader(
		train_ds,
		batch_size=config.batch_size,
		shuffle=False,
		drop_last=False,
	)

	for epoch in range(1, config.max_epochs + 1):
		model.train()
		
		# Training loop with oversampling
		for xb, sb, yb in train_loader:
			xb = xb.to(device)
			sb = sb.to(device)
			yb = yb.to(device)
			optimizer.zero_grad()
			logits = model(xb, sb)
			loss = criterion(logits, yb)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
		
		# Compute training loss on original (non-oversampled) data
		model.eval()
		train_loss_total = 0.0
		train_count = 0
		with torch.no_grad():
			for xb, sb, yb in train_loader_eval:
				xb = xb.to(device)
				sb = sb.to(device)
				yb = yb.to(device)
				logits = model(xb, sb)
				batch_loss = criterion(logits, yb)
				train_loss_total += batch_loss.item() * xb.size(0)
				train_count += xb.size(0)
		avg_train_loss = train_loss_total / train_count
		model.train()  # Set back to train mode

		# Evaluate: get both window and subject level results
		# First get window-level predictions and loss
		model.eval()
		all_logits = []
		all_labels = []
		val_loss_total = 0.0
		val_loss_unweighted_total = 0.0  # Track unweighted loss for monitoring
		val_count = 0
		# Create unweighted criterion for monitoring (without class weights or focal loss)
		unweighted_criterion = nn.CrossEntropyLoss()
		with torch.no_grad():
			for xb, sb, yb in val_loader:
				xb = xb.to(device)
				sb = sb.to(device)
				yb = yb.to(device)
				logits = model(xb, sb)
				# Compute validation loss using the same criterion (weighted/focal)
				batch_loss = criterion(logits, yb)
				val_loss_total += batch_loss.item() * xb.size(0)
				# Also compute unweighted loss for monitoring
				batch_loss_unweighted = unweighted_criterion(logits, yb)
				val_loss_unweighted_total += batch_loss_unweighted.item() * xb.size(0)
				val_count += xb.size(0)
				all_logits.append(logits.cpu().numpy())
				all_labels.append(yb.cpu().numpy())
		all_logits = np.concatenate(all_logits, axis=0)
		all_labels = np.concatenate(all_labels, axis=0)
		val_win_probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
		avg_val_loss = val_loss_total / val_count
		avg_val_loss_unweighted = val_loss_unweighted_total / val_count
		
		# Aggregate to subject level for F1 calculation
		val_subj_probs = aggregate_subject_probs(val_win_probs, val_window_owners.tolist(), num_val_subjects)
		subject_labels = np.zeros(num_val_subjects, dtype=np.int64)
		for subj_idx in range(num_val_subjects):
			win_mask = val_window_owners == subj_idx
			if win_mask.any():
				subject_labels[subj_idx] = all_labels[win_mask][0]
		
		# Use simple argmax for F1 calculation during training (no threshold tuning)
		# Threshold tuning will be done at the end only
		subject_preds = val_subj_probs.argmax(axis=1)
		val_f1 = f1_score(subject_labels, subject_preds, average="macro")
		
		current_lr = optimizer.param_groups[0]["lr"]
		lr_str = f"{current_lr:.6f}" if current_lr >= 1e-6 else f"{current_lr:.2e}"
		print(f"[Fold {fold}] Epoch {epoch:02d} - lr: {lr_str}, train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f} (unweighted: {avg_val_loss_unweighted:.4f}), val_macro_f1: {val_f1:.4f}")

		# Track best F1 (for model selection)
		f1_improved = val_f1 > best_f1 + 1e-4
		if f1_improved:
			best_f1 = val_f1
			epochs_no_improve_f1 = 0
			# Save model when F1 improves
			best_state = {
				"model_state": model.state_dict(),
				"mean": mean,
				"std": std,
				"val_win_probs": val_win_probs,
			}
		else:
			epochs_no_improve_f1 += 1
		
		# Track best validation loss (for monitoring only)
		loss_improved = avg_val_loss < best_val_loss - 1e-4
		if loss_improved:
			best_val_loss = avg_val_loss
			epochs_no_improve_loss = 0
		else:
			epochs_no_improve_loss += 1
		
		# Update learning rate scheduler based on F1 score
		if epoch < config.warmup_epochs:
			warmup_lr = config.lr * (epoch + 1) / config.warmup_epochs
			for param_group in optimizer.param_groups:
				param_group["lr"] = warmup_lr
		else:
			# Reduce LR if F1 plateaus (ReduceLROnPlateau)
			scheduler.step(val_f1)
		
		# Early stopping: Use F1 score (the metric we care about) instead of loss
		if epochs_no_improve_f1 >= config.patience:
			print(f"Early stopping: validation F1 hasn't improved for {config.patience} epochs (best: {best_f1:.4f})")
			break

	if best_state is None:
		# Fallback: evaluate one more time
		_, val_win_probs, _ = evaluate(model, val_loader, device)
		best_state = {
			"model_state": model.state_dict(),
			"mean": mean,
			"std": std,
			"val_win_probs": val_win_probs,
		}
	
	# Do threshold tuning once at the end using best validation predictions
	val_subj_probs = aggregate_subject_probs(best_state["val_win_probs"], val_window_owners.tolist(), num_val_subjects)
	subject_labels = np.zeros(num_val_subjects, dtype=np.int64)
	for subj_idx in range(num_val_subjects):
		win_mask = val_window_owners == subj_idx
		if win_mask.any():
			subject_labels[subj_idx] = val_labels[subj_idx]
	optimal_thresholds = find_optimal_thresholds(val_subj_probs, subject_labels)
	best_state["optimal_thresholds"] = optimal_thresholds

	return best_state["val_win_probs"], best_state


def make_windows(
	sequences: np.ndarray,
	static_feats: np.ndarray,
	window_size: int = 64,
	stride: int = 16,
	target_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
	"""
	Produce overlapping windows and carry static features for each window. Windows are zero-padded
	to `window_size` and include an additional mask channel indicating valid timesteps.
	Returns:
	 - win_seq: [num_windows, window_size, features + 1] (last channel is mask)
	 - win_static: [num_windows, static_dim]
	 - owner_indices: list mapping each window to its subject index (0..N-1)
	"""
	N, T, F = sequences.shape
	if target_len is None:
		target_len = window_size
	windows: List[np.ndarray] = []
	wstatic: List[np.ndarray] = []
	owners: List[int] = []

	for i in range(N):
		starts: List[int] = []
		t = 0
		while t + window_size <= T:
			starts.append(t)
			t += stride
		if not starts or starts[-1] != T - window_size:
			starts.append(T - window_size)
		for start in starts:
			data = np.zeros((target_len, F), dtype=np.float32)
			valid_segment = sequences[i, start:start + window_size]
			data[:valid_segment.shape[0]] = valid_segment
			mask = np.zeros((target_len, 1), dtype=np.float32)
			mask[:valid_segment.shape[0]] = 1.0
			window = np.concatenate([data, mask], axis=1)
			windows.append(window)
			wstatic.append(static_feats[i])
			owners.append(i)

	return np.stack(windows, axis=0), np.stack(wstatic, axis=0), owners


def aggregate_subject_probs(
	win_probs: np.ndarray,
	win_owners: List[int],
	num_subjects: int,
) -> np.ndarray:
	"""
	Aggregate per-window probabilities into per-subject probabilities by averaging.
	"""
	out = np.zeros((num_subjects, len(LABELS)), dtype=np.float32)
	counts = np.zeros(num_subjects, dtype=np.int32)
	for p, owner in zip(win_probs, win_owners):
		out[owner] += p
		counts[owner] += 1
	counts[counts == 0] = 1
	out = out / counts[:, None]
	return out


def main():
	set_seed(42)
	device = detect_device()
	print(f"Using device: {device}")

	X_train_df, y_series, _ = load_frames()
	sample_ids = y_series.index.tolist()
	y_array = y_series.values.astype(np.int64)

	print("Preparing sequences...")
	raw_sequences, static_feats, feature_names = create_raw_sequences(X_train_df, sample_ids)
	print(f"Sequence tensor shape: {raw_sequences.shape} (samples, time, features={len(feature_names)})")

	config = TrainConfig(input_dim=raw_sequences.shape[-1] + 1, static_dim=static_feats.shape[1])

	# Windowing parameters
	max_window_len = max(config.window_sizes)

	# Log dataset information
	print("\n" + "="*60)
	print("DATASET INFORMATION")
	print("="*60)
	print(f"Total samples: {len(sample_ids)}")
	print(f"Sequence length: {raw_sequences.shape[1]} timesteps")
	print(f"Temporal features: {raw_sequences.shape[2]}")
	print(f"Static features: {static_feats.shape[1]}")
	print(f"Total input features (with mask): {config.input_dim}")
	
	# Class distribution
	class_counts = np.bincount(y_array, minlength=len(LABELS))
	class_dist = {LABELS[i]: int(count) for i, count in enumerate(class_counts)}
	print(f"\nClass distribution:")
	for label, count in class_dist.items():
		pct = 100.0 * count / len(y_array)
		print(f"  {label:12s}: {count:4d} ({pct:5.2f}%)")
	
	# Compute expected windows per sample (matches make_windows logic exactly)
	seq_len = raw_sequences.shape[1]
	def count_windows(win_size: int, stride: int, seq_len: int) -> int:
		starts = []
		t = 0
		while t + win_size <= seq_len:
			starts.append(t)
			t += stride
		# Add final window if needed (matches make_windows: if not starts or starts[-1] != T - window_size)
		if not starts or starts[-1] != seq_len - win_size:
			starts.append(seq_len - win_size)
		return len(starts)
	
	windows_per_size = [count_windows(win_size, config.stride, seq_len) for win_size in config.window_sizes]
	total_windows_per_sample = sum(windows_per_size)
	
	print(f"\nWindowing configuration:")
	print(f"  Window sizes: {config.window_sizes}")
	print(f"  Stride: {config.stride}")
	print(f"  Max window length: {max_window_len}")
	for win_size, win_count in zip(config.window_sizes, windows_per_size):
		print(f"    Windows (size={win_size}): {win_count} per sample")
	print(f"  Total windows per sample: {total_windows_per_sample}")
	print(f"  Expected total windows: ~{len(sample_ids) * total_windows_per_sample:,}")

	# Log model information
	print("\n" + "="*60)
	print("MODEL INFORMATION")
	print("="*60)
	# Create a temporary model to count parameters
	temp_model = GRUClassifier(config, num_classes=len(LABELS)).to(device)
	total_params = sum(p.numel() for p in temp_model.parameters())
	trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
	direction_factor = 2 if config.bidirectional else 1
	
	print(f"Architecture: GRU Classifier")
	print(f"  GRU layers: {config.num_layers}")
	print(f"  Hidden dimension: {config.hidden_dim}")
	print(f"  Bidirectional: {config.bidirectional}")
	print(f"  GRU output dim: {config.hidden_dim * direction_factor}")
	print(f"  Attention: {config.use_attention}")
	if config.use_attention:
		print(f"    -> Uses attention over all timesteps")
	else:
		print(f"    -> Uses last hidden state")
	print(f"  Static MLP: {config.static_dim} -> 64 -> 32")
	print(f"  Final FC: {config.hidden_dim * direction_factor + 32} -> 128 -> {len(LABELS)}")
	print(f"  Dropout: {config.dropout}")
	print(f"\nParameters:")
	print(f"  Total: {total_params:,}")
	print(f"  Trainable: {trainable_params:,}")
	
	# Clean up temporary model
	del temp_model
	try:
		if device.type == "cuda":
			torch.cuda.empty_cache()
		elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
			torch.mps.empty_cache()
	except:
		pass
	
	# Log training configuration
	print("\n" + "="*60)
	print("TRAINING CONFIGURATION")
	print("="*60)
	print(f"Optimizer: AdamW")
	print(f"  Learning rate: {config.lr}")
	print(f"  Weight decay: {config.weight_decay}")
	print(f"  LR schedule: CosineAnnealing (warmup={config.warmup_epochs} epochs)")
	print(f"Batch size: {config.batch_size}")
	print(f"Max epochs: {config.max_epochs}")
	print(f"Early stopping:")
	print(f"  F1 patience: {config.patience} epochs")
	print(f"  Loss patience: {config.patience_loss} epochs")
	if config.use_focal_loss:
		print(f"Loss: Focal Loss (gamma={config.focal_gamma}) with class weights")
	else:
		print(f"Loss: CrossEntropyLoss with class weights (inverse frequency)")
		print(f"  Label smoothing: {config.label_smoothing}")
	print(f"Data augmentation:")
	print(f"  Jitter std: {config.jitter_std}")
	print(f"  Time mask prob: {config.time_mask_prob} (max length: {config.time_mask_max})")
	print(f"  Time shift max: {config.time_shift_max}")
	print(f"Oversampling: WeightedRandomSampler (inverse frequency)")
	print(f"Evaluation: Subject-level aggregation (window -> subject)")
	print("="*60 + "\n")

	gkf = GroupKFold(n_splits=5)
	oof_probs = np.zeros((len(sample_ids), len(LABELS)), dtype=np.float32)
	fold_reports = []

	for fold, (train_idx, val_idx) in enumerate(gkf.split(sample_ids, y_array, groups=sample_ids), start=1):
		# Build windows separately for train and val
		train_seq = raw_sequences[train_idx]
		train_static = static_feats[train_idx]
		train_labels = y_array[train_idx]

		val_seq = raw_sequences[val_idx]
		val_static = static_feats[val_idx]
		val_labels = y_array[val_idx]

		train_win_seqs = []
		train_win_static = []
		train_owners = []
		val_win_seqs = []
		val_win_static = []
		val_owners = []

		for win_size in config.window_sizes:
			t_seq, t_stat, t_owner = make_windows(
				train_seq,
				train_static,
				window_size=win_size,
				stride=config.stride,
				target_len=max_window_len,
			)
			v_seq, v_stat, v_owner = make_windows(
				val_seq,
				val_static,
				window_size=win_size,
				stride=config.stride,
				target_len=max_window_len,
			)
			train_win_seqs.append(t_seq)
			train_win_static.append(t_stat)
			train_owners.extend(t_owner)
			val_win_seqs.append(v_seq)
			val_win_static.append(v_stat)
			val_owners.extend(v_owner)

		train_win_seq = np.concatenate(train_win_seqs, axis=0)
		train_win_static = np.concatenate(train_win_static, axis=0)
		val_win_seq = np.concatenate(val_win_seqs, axis=0)
		val_win_static = np.concatenate(val_win_static, axis=0)

		train_owners = np.array(train_owners, dtype=np.int64)
		val_owners = np.array(val_owners, dtype=np.int64)

		# Window labels: inherit subject labels
		train_win_labels = train_labels[train_owners]
		val_win_labels = val_labels[val_owners]

		print(f"===== Fold {fold} / 5 =====")
		print(f"Train windows: {len(train_win_labels)}, Val windows: {len(val_win_labels)}")
		print(f"Train class dist: {np.bincount(train_win_labels, minlength=len(LABELS))}")
		print(f"Val class dist: {np.bincount(val_win_labels, minlength=len(LABELS))}")
		
		val_win_probs, best_state = train_one_fold(
			fold=fold,
			config=config,
			train_seq=train_win_seq,
			train_static=train_win_static,
			train_labels=train_win_labels,
			val_seq=val_win_seq,
			val_static=val_win_static,
			val_labels=val_win_labels,
			val_window_owners=val_owners,
			num_val_subjects=len(val_idx),
			device=device,
		)

		# Aggregate window probs back to subject level
		val_probs = aggregate_subject_probs(val_win_probs, val_owners, num_subjects=len(val_idx))
		
		# Apply optimal thresholds if available
		if "optimal_thresholds" in best_state:
			optimal_thresholds = best_state["optimal_thresholds"]
			val_probs_adjusted = val_probs * optimal_thresholds
			val_probs_adjusted = val_probs_adjusted / val_probs_adjusted.sum(axis=1, keepdims=True)
			val_preds = val_probs_adjusted.argmax(axis=1)
			oof_probs[val_idx] = val_probs_adjusted
		else:
			val_preds = val_probs.argmax(axis=1)
			oof_probs[val_idx] = val_probs
		
		fold_f1 = f1_score(val_labels, val_preds, average="macro")
		report = classification_report(val_labels, val_preds, target_names=LABELS, digits=4)
		print(f"[Fold {fold}] Macro-F1: {fold_f1:.4f}")
		if "optimal_thresholds" in best_state:
			print(f"[Fold {fold}] Optimal thresholds: {best_state['optimal_thresholds']}")
		print(report)
		fold_reports.append({"fold": fold, "macro_f1": fold_f1, "report": report})

	oof_preds = oof_probs.argmax(axis=1)
	macro_f1 = f1_score(y_array, oof_preds, average="macro")
	print("===== CV Summary =====")
	print(f"OOF Macro-F1: {macro_f1:.5f}")
	print(classification_report(y_array, oof_preds, target_names=LABELS, digits=4))

	out_path = os.path.join(OUT_DIR, "gru_cv_metrics.json")
	with open(out_path, "w") as f:
		json.dump(
			{
				"oof_macro_f1": macro_f1,
				"folds": [{"fold": fr["fold"], "macro_f1": fr["macro_f1"]} for fr in fold_reports],
			},
			f,
			indent=2,
		)
	print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
	main()


