#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from pathlib import Path
import logging
import sys
from typing import Dict, Any, Optional
import shutil

# Optional: Weights & Biases integration
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print(
        "Warning: wandb not installed. Install with 'pip install wandb' for experiment tracking."
    )


# ----------------- Logging Setup -----------------
def setup_logger(log_dir: str, name: str = "training") -> logging.Logger:
    """Setup comprehensive logging to file and console"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ----------------- Experiment Tracker -----------------
class ExperimentTracker:
    """Unified experiment tracking for multiple backends"""

    def __init__(
        self,
        run_dir: str,
        config: Dict[str, Any],
        use_wandb: bool = True,
        project_name: str = "two-stage-column-mapping",
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.metrics_log = []

        # Initialize wandb if available
        if self.use_wandb:
            try:
                wandb.init(
                    project=project_name,
                    config=config,
                    dir=str(self.run_dir),
                    name=self.run_dir.name,
                )
                print(f"✓ Wandb initialized for experiment: {wandb.run.name}")
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                self.use_wandb = False

        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        stage: str = "train",
        epoch: Optional[int] = None,
    ):
        """Log metrics to all backends"""
        # Add timestamp and metadata
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "epoch": epoch,
            "step": step,
            **metrics,
        }
        self.metrics_log.append(log_entry)

        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {f"{stage}/{k}": v for k, v in metrics.items()}
            if epoch is not None:
                wandb_metrics["epoch"] = epoch
            wandb.log(wandb_metrics, step=step)

        # Save to CSV incrementally
        self._save_metrics_csv()

    def _save_metrics_csv(self):
        """Save metrics to CSV"""
        if self.metrics_log:
            df = pd.DataFrame(self.metrics_log)
            df.to_csv(self.run_dir / "metrics.csv", index=False)

    def log_artifact(self, file_path: str, artifact_type: str = "model"):
        """Log artifact to wandb"""
        if self.use_wandb:
            try:
                artifact = wandb.Artifact(
                    name=f"{artifact_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    type=artifact_type,
                )
                artifact.add_file(file_path)
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Warning: Failed to log artifact: {e}")

    def log_confusion_matrix(
        self, cm: np.ndarray, class_names: list, stage: str, epoch: int
    ):
        """Log confusion matrix as image"""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title(f"{stage} Confusion Matrix - Epoch {epoch}")

        # Save locally
        cm_path = self.run_dir / f"{stage}_confusion_matrix_epoch_{epoch}.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Log to wandb
        if self.use_wandb:
            wandb.log(
                {f"{stage}/confusion_matrix": wandb.Image(str(cm_path))}, step=epoch
            )

    def save_training_history(
        self, history: Dict[str, Any], filename: str = "training_history.json"
    ):
        """Save training history to JSON"""
        with open(self.run_dir / filename, "w") as f:
            json.dump(history, f, indent=2)

    def finish(self):
        """Cleanup and finalize experiment"""
        self._save_metrics_csv()
        if self.use_wandb:
            wandb.finish()


# ----------------- Checkpoint Manager -----------------
class CheckpointManager:
    """Manage model checkpoints with versioning"""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        self.best_metric = None
        self.best_checkpoint_path = None

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        epoch: int,
        metric_value: Optional[float] = None,
        is_best: bool = False,
        stage: str = "stage1",
    ):
        """Save checkpoint with metadata"""
        checkpoint_name = f"{stage}_epoch_{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Add metadata
        state["epoch"] = epoch
        state["timestamp"] = datetime.now().isoformat()
        if metric_value is not None:
            state["metric_value"] = metric_value

        # Save checkpoint
        torch.save(state, checkpoint_path)
        self.checkpoints.append(
            {
                "path": checkpoint_path,
                "epoch": epoch,
                "metric": metric_value,
                "timestamp": state["timestamp"],
            }
        )

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / f"{stage}_best.pt"
            shutil.copy(checkpoint_path, best_path)
            self.best_checkpoint_path = best_path
            self.best_metric = metric_value
            print(f"✓ Saved best {stage} model (metric: {metric_value:.4f})")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(stage)

        # Save checkpoint manifest
        self._save_manifest()

        return checkpoint_path

    def _cleanup_old_checkpoints(self, stage: str):
        """Remove old checkpoints keeping only max_checkpoints"""
        stage_checkpoints = [
            cp
            for cp in self.checkpoints
            if stage in cp["path"].name and "best" not in cp["path"].name
        ]

        if len(stage_checkpoints) > self.max_checkpoints:
            # Sort by epoch (oldest first)
            stage_checkpoints.sort(key=lambda x: x["epoch"])

            # Remove oldest
            for cp in stage_checkpoints[: -self.max_checkpoints]:
                if cp["path"].exists():
                    cp["path"].unlink()
                self.checkpoints.remove(cp)

    def _save_manifest(self):
        """Save checkpoint manifest"""
        manifest = {
            "checkpoints": [
                {
                    "path": str(cp["path"].name),
                    "epoch": cp["epoch"],
                    "metric": cp["metric"],
                    "timestamp": cp["timestamp"],
                }
                for cp in self.checkpoints
            ],
            "best_metric": self.best_metric,
            "best_checkpoint": (
                str(self.best_checkpoint_path.name)
                if self.best_checkpoint_path
                else None
            ),
        }

        with open(self.checkpoint_dir / "checkpoint_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def load_checkpoint(
        self, checkpoint_path: str, device: str = "cuda"
    ) -> Dict[str, Any]:
        """Load checkpoint from path"""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return torch.load(checkpoint_path, map_location=device, weights_only=False)

    def get_latest_checkpoint(self, stage: str) -> Optional[Path]:
        """Get path to latest checkpoint for stage"""
        stage_checkpoints = [cp for cp in self.checkpoints if stage in cp["path"].name]
        if stage_checkpoints:
            return max(stage_checkpoints, key=lambda x: x["epoch"])["path"]
        return None


# ----------------- Early Stopping -----------------
class EarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(
                    f"\n⚠ Early stopping triggered after {self.counter} epochs without improvement"
                )
                print(
                    f"   Best score: {self.best_score:.4f} at epoch {self.best_epoch}"
                )
                return True

        return False


# ----------------- Reproducibility helper -----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============== STAGE 1 MODEL: File Name → Table Classification ==============


class FileToTableDataset(Dataset):
    """Dataset for Stage 1: File name to table classification"""

    def __init__(self, df, tokenizer, max_length=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create label mapping
        self.tables = sorted(df["table"].unique())
        self.table_to_idx = {table: idx for idx, table in enumerate(self.tables)}
        self.idx_to_table = {idx: table for table, idx in self.table_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Tokenize file name
        encoding = self.tokenizer(
            row["file_name"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.table_to_idx[row["table"]], dtype=torch.long),
        }


class FileToTableClassifier(nn.Module):
    """Stage 1 Model: Classify file name to table"""

    def __init__(
        self,
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        num_tables=5,
        hidden_dim=256,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.encoder_dim = getattr(self.encoder.config, "hidden_size", 384)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tables),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits


# ============== STAGE 2 MODEL: Column Mapping Within Table ==============


class ColumnMappingDataset(Dataset):
    """Dataset for Stage 2: Column mapping within table context"""

    def __init__(self, df, tokenizer, max_length=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Tokenize raw column
        raw_encoding = self.tokenizer(
            row["raw_column"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize standard column with table context
        standard_with_context = f"[{row['table_context']}] {row['standard_column']}"
        standard_encoding = self.tokenizer(
            standard_with_context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "raw_input_ids": raw_encoding["input_ids"].squeeze(0),
            "raw_attention_mask": raw_encoding["attention_mask"].squeeze(0),
            "standard_input_ids": standard_encoding["input_ids"].squeeze(0),
            "standard_attention_mask": standard_encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.float32),
        }


class SiameseColumnMapper(nn.Module):
    """Stage 2 Model: Siamese network for column mapping"""

    def __init__(
        self,
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.encoder_dim = getattr(self.encoder.config, "hidden_size", 384)

        self.similarity_features = nn.Sequential(
            nn.Linear(self.encoder_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        raw_input_ids,
        raw_attention_mask,
        standard_input_ids,
        standard_attention_mask,
    ):
        raw_embedding = self.encode_text(raw_input_ids, raw_attention_mask)
        standard_embedding = self.encode_text(
            standard_input_ids, standard_attention_mask
        )

        similarity_vector = self.compute_similarity_features(
            raw_embedding, standard_embedding
        )

        features = self.similarity_features(similarity_vector)
        logits = self.classifier(features)
        return torch.sigmoid(logits)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def compute_similarity_features(self, emb1, emb2):
        return torch.cat([emb1, emb2, torch.abs(emb1 - emb2), emb1 * emb2], dim=1)


# ============== TRAINING UTILITIES ==============


class TemperatureScaling(nn.Module):
    """Temperature scaling for confidence calibration"""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, logits):
        return logits / self.temperature


def calibrate_model_stage1(model, calibration_loader, device, logger):
    """Calibrate Stage 1 model using temperature scaling"""
    temperature_model = TemperatureScaling().to(device)
    optimizer = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)

    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in calibration_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    def eval_loss():
        optimizer.zero_grad()
        scaled = temperature_model(all_logits)
        loss = nn.CrossEntropyLoss()(scaled, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    t = float(temperature_model.temperature.item())
    logger.info(f"Optimal temperature (stage1): {t:.4f}")
    return t


def calibrate_model_stage2(model, calibration_loader, device, logger):
    """Calibrate Stage 2 model using temperature scaling"""
    temperature_model = TemperatureScaling().to(device)
    optimizer = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)

    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in calibration_loader:
            raw_input_ids = batch["raw_input_ids"].to(device)
            raw_attention_mask = batch["raw_attention_mask"].to(device)
            standard_input_ids = batch["standard_input_ids"].to(device)
            standard_attention_mask = batch["standard_attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                raw_input_ids,
                raw_attention_mask,
                standard_input_ids,
                standard_attention_mask,
            )
            all_logits.append(outputs)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits).squeeze()
    all_labels = torch.cat(all_labels)

    all_logits = torch.logit(all_logits.clamp(1e-7, 1 - 1e-7))

    def eval_loss():
        optimizer.zero_grad()
        calibrated = torch.sigmoid(temperature_model(all_logits))
        loss = nn.BCELoss()(calibrated, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    t = float(temperature_model.temperature.item())
    logger.info(f"Optimal temperature (stage2): {t:.4f}")
    return t


# ============== STAGE 1 TRAINING ==============


def train_stage1(
    train_loader,
    val_loader,
    num_tables,
    tracker: ExperimentTracker,
    checkpoint_manager: CheckpointManager,
    logger: logging.Logger,
    num_epochs=10,
    learning_rate=2e-5,
    device="cuda",
    early_stopping_patience=5,
    resume_from=None,
):
    """Train Stage 1 model with full production features"""

    logger.info("=" * 80)
    logger.info("TRAINING STAGE 1: File Name → Table Classification")
    logger.info("=" * 80)

    model = FileToTableClassifier(num_tables=num_tables).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    start_epoch = 0
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_top3_acc": [],
    }

    # Resume from checkpoint if specified
    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = checkpoint_manager.load_checkpoint(resume_from, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        training_history = checkpoint.get("training_history", training_history)
        logger.info(f"Resumed from epoch {start_epoch}")

    early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")
    best_val_acc = (
        max(training_history["val_acc"]) if training_history["val_acc"] else 0.0
    )

    global_step = 0

    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({"loss": loss.item(), "acc": train_correct / train_total})

            # Log batch metrics
            if batch_idx % 10 == 0:
                tracker.log_metrics(
                    {
                        "batch_loss": loss.item(),
                        "batch_acc": train_correct / train_total,
                    },
                    step=global_step,
                    stage="stage1_train",
                    epoch=epoch,
                )

            global_step += 1

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_acc = train_correct / max(1, train_total)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0

        # Top-3 accuracy
        all_probs = np.array(all_probs)
        k = min(3, all_probs.shape[1]) if all_probs.size else 0
        if k > 0:
            topk_preds = np.argsort(all_probs, axis=1)[:, -k:]
            top3_acc = (
                np.mean([label in topk_preds[i] for i, label in enumerate(all_labels)])
                if len(all_labels) > 0
                else 0.0
            )
        else:
            top3_acc = 0.0

        # Update history
        training_history["train_loss"].append(avg_train_loss)
        training_history["val_loss"].append(avg_val_loss)
        training_history["val_acc"].append(val_acc)
        training_history["val_top3_acc"].append(top3_acc)

        # Log metrics
        epoch_metrics = {
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_top3_acc": top3_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        tracker.log_metrics(epoch_metrics, step=epoch, stage="stage1", epoch=epoch)

        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Top-3: {top3_acc:.4f}"
        )

        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_history": training_history,
            "config": tracker.config,
        }

        checkpoint_manager.save_checkpoint(
            checkpoint_state, epoch, val_acc, is_best, stage="stage1"
        )

        # Early stopping check
        if early_stopping(val_acc, epoch):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

        logger.info("-" * 80)

    # Load best model
    if checkpoint_manager.best_checkpoint_path:
        logger.info(
            f"Loading best model from {checkpoint_manager.best_checkpoint_path}"
        )
        best_checkpoint = checkpoint_manager.load_checkpoint(
            str(checkpoint_manager.best_checkpoint_path), device
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])

    return model, training_history


# ============== STAGE 2 TRAINING ==============


def train_stage2(
    train_loader,
    val_loader,
    tracker: ExperimentTracker,
    checkpoint_manager: CheckpointManager,
    logger: logging.Logger,
    num_epochs=10,
    learning_rate=2e-5,
    device="cuda",
    early_stopping_patience=5,
    resume_from=None,
):
    """Train Stage 2 model with full production features"""

    logger.info("=" * 80)
    logger.info("TRAINING STAGE 2: Column Mapping Within Table")
    logger.info("=" * 80)

    model = SiameseColumnMapper().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    start_epoch = 0
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    # Resume from checkpoint if specified
    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = checkpoint_manager.load_checkpoint(resume_from, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        training_history = checkpoint.get("training_history", training_history)
        logger.info(f"Resumed from epoch {start_epoch}")

    early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")
    best_val_f1 = max(training_history["val_f1"]) if training_history["val_f1"] else 0.0

    global_step = 0

    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            raw_input_ids = batch["raw_input_ids"].to(device)
            raw_attention_mask = batch["raw_attention_mask"].to(device)
            standard_input_ids = batch["standard_input_ids"].to(device)
            standard_attention_mask = batch["standard_attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(
                raw_input_ids,
                raw_attention_mask,
                standard_input_ids,
                standard_attention_mask,
            )
            loss = criterion(outputs.flatten(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs.flatten() > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({"loss": loss.item(), "acc": train_correct / train_total})

            # Log batch metrics
            if batch_idx % 10 == 0:
                tracker.log_metrics(
                    {
                        "batch_loss": loss.item(),
                        "batch_acc": train_correct / train_total,
                    },
                    step=global_step,
                    stage="stage2_train",
                    epoch=epoch,
                )

            global_step += 1

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_acc = train_correct / max(1, train_total)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                raw_input_ids = batch["raw_input_ids"].to(device)
                raw_attention_mask = batch["raw_attention_mask"].to(device)
                standard_input_ids = batch["standard_input_ids"].to(device)
                standard_attention_mask = batch["standard_attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(
                    raw_input_ids,
                    raw_attention_mask,
                    standard_input_ids,
                    standard_attention_mask,
                )
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

                probs = outputs.squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        avg_val_loss = val_loss / max(1, len(val_loader))

        # Calculate metrics
        if len(all_labels) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="binary", zero_division=0
            )
            accuracy = accuracy_score(all_labels, all_preds)
        else:
            precision = recall = f1 = accuracy = 0.0

        # Update history
        training_history["train_loss"].append(avg_train_loss)
        training_history["val_loss"].append(avg_val_loss)
        training_history["val_precision"].append(precision)
        training_history["val_recall"].append(recall)
        training_history["val_f1"].append(f1)

        # Log metrics
        epoch_metrics = {
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_acc": accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        tracker.log_metrics(epoch_metrics, step=epoch, stage="stage2", epoch=epoch)

        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Val P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")

        # Save checkpoint
        is_best = f1 > best_val_f1
        if is_best:
            best_val_f1 = f1

        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_history": training_history,
            "config": tracker.config,
        }

        checkpoint_manager.save_checkpoint(
            checkpoint_state, epoch, f1, is_best, stage="stage2"
        )

        # Early stopping check
        if early_stopping(f1, epoch):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

        logger.info("-" * 80)

    # Load best model
    if checkpoint_manager.best_checkpoint_path:
        logger.info(
            f"Loading best model from {checkpoint_manager.best_checkpoint_path}"
        )
        best_checkpoint = checkpoint_manager.load_checkpoint(
            str(checkpoint_manager.best_checkpoint_path), device
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])

    return model, training_history


# ============== EVALUATION ==============


def evaluate_stage1(model, test_loader, dataset, device, logger, tracker, epoch=None):
    """Comprehensive evaluation of Stage 1 model"""
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING STAGE 1 MODEL")
    logger.info("=" * 80)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0

    # Top-3 accuracy
    all_probs = np.array(all_probs)
    if all_probs.size:
        topk = min(3, all_probs.shape[1])
        topk_preds = np.argsort(all_probs, axis=1)[:, -topk:]
        top3_acc = (
            np.mean([label in topk_preds[i] for i, label in enumerate(all_labels)])
            if len(all_labels) > 0
            else 0.0
        )
    else:
        top3_acc = 0.0

    logger.info(f"\nTop-1 Accuracy: {accuracy:.4f}")
    logger.info(f"Top-3 Accuracy: {top3_acc:.4f}")

    # Confusion matrix
    cm = (
        confusion_matrix(all_labels, all_preds)
        if len(all_labels) > 0
        else np.array([[]])
    )

    # Per-class accuracy
    logger.info("\nPer-Table Accuracy:")
    if cm.size:
        for i, table in enumerate(dataset.tables):
            denom = cm[i].sum() if i < cm.shape[0] else 0
            table_acc = cm[i, i] / denom if denom > 0 else 0
            logger.info(f"  {table}: {table_acc:.4f}")

        # Log confusion matrix to tracker
        if epoch is not None:
            tracker.log_confusion_matrix(cm, dataset.tables, "stage1_eval", epoch)
    else:
        logger.info("  No samples in test set.")

    # Log evaluation metrics
    eval_metrics = {"eval_accuracy": accuracy, "eval_top3_accuracy": top3_acc}
    tracker.log_metrics(eval_metrics, stage="stage1_eval", epoch=epoch)

    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_acc,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def evaluate_stage2(model, test_loader, device, logger, tracker, epoch=None):
    """Comprehensive evaluation of Stage 2 model"""
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING STAGE 2 MODEL")
    logger.info("=" * 80)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            raw_input_ids = batch["raw_input_ids"].to(device)
            raw_attention_mask = batch["raw_attention_mask"].to(device)
            standard_input_ids = batch["standard_input_ids"].to(device)
            standard_attention_mask = batch["standard_attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                raw_input_ids,
                raw_attention_mask,
                standard_input_ids,
                standard_attention_mask,
            )
            probs = outputs.flatten().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    # Calculate metrics
    if len(all_labels) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", zero_division=0
        )
        accuracy = accuracy_score(all_labels, all_preds)
    else:
        precision = recall = f1 = accuracy = 0.0

    logger.info(f"\nAccuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Confusion matrix
    cm = (
        confusion_matrix(all_labels, all_preds)
        if len(all_labels) > 0
        else np.array([[]])
    )
    logger.info(f"\nConfusion Matrix:")
    logger.info(cm)

    # Log confusion matrix to tracker
    if epoch is not None and cm.size:
        tracker.log_confusion_matrix(cm, ["Negative", "Positive"], "stage2_eval", epoch)

    # Log evaluation metrics
    eval_metrics = {
        "eval_accuracy": accuracy,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
    }
    tracker.log_metrics(eval_metrics, stage="stage2_eval", epoch=epoch)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


# ============== MAIN TRAINING PIPELINE ==============


def main():
    set_seed(42)

    # Configuration
    config = {
        "data_dir": "./two_stage_datasets",
        "save_dir": "./runs",
        "batch_size": 32,
        "num_epochs_stage1": 10,
        "num_epochs_stage2": 10,
        "learning_rate": 2e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_workers": 4,
        "early_stopping_patience": 5,
        "max_checkpoints": 5,
        "use_wandb": True,
        "wandb_project": "two-stage-column-mapping",
        "resume_stage1": "./runs/run_20251002_164023/checkpoints/stage1_epoch_9.pt",  # Path to checkpoint to resume from
        "resume_stage2": "./runs/run_20251002_164023/checkpoints/stage2_epoch_8.pt",
    }

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["save_dir"]) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(str(run_dir))
    logger.info("=" * 80)
    logger.info("PRODUCTION TWO-STAGE COLUMN MAPPING TRAINING")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"\nConfiguration:\n{json.dumps(config, indent=2)}")

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        run_dir=str(run_dir),
        config=config,
        use_wandb=config["use_wandb"],
        project_name=config["wandb_project"],
    )

    # Initialize checkpoint manager
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir), max_checkpoints=config["max_checkpoints"]
    )

    # Load tokenizer
    logger.info(f"\nLoading tokenizer: {config['encoder_model']}")
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_model"])

    # ========== STAGE 1 ==========
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: FILE NAME → TABLE CLASSIFICATION")
    logger.info("=" * 80)

    # Load Stage 1 data
    logger.info("Loading Stage 1 datasets...")
    stage1_train_df = pd.read_csv(f"{config['data_dir']}/stage1_train.csv")
    stage1_val_df = pd.read_csv(f"{config['data_dir']}/stage1_val.csv")
    stage1_cal_df = pd.read_csv(f"{config['data_dir']}/stage1_calibration.csv")

    logger.info(f"Stage 1 Train samples: {len(stage1_train_df)}")
    logger.info(f"Stage 1 Val samples: {len(stage1_val_df)}")
    logger.info(f"Stage 1 Cal samples: {len(stage1_cal_df)}")

    # Create datasets
    stage1_train_dataset = FileToTableDataset(stage1_train_df, tokenizer)
    stage1_val_dataset = FileToTableDataset(stage1_val_df, tokenizer)
    stage1_cal_dataset = FileToTableDataset(stage1_cal_df, tokenizer)

    num_tables = len(stage1_train_dataset.tables)
    logger.info(f"Number of tables: {num_tables}")
    logger.info(f"Tables: {stage1_train_dataset.tables}")

    # Create dataloaders
    stage1_train_loader = DataLoader(
        stage1_train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    stage1_val_loader = DataLoader(
        stage1_val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    stage1_cal_loader = DataLoader(
        stage1_cal_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # Train Stage 1
    stage1_model, stage1_history = train_stage1(
        stage1_train_loader,
        stage1_val_loader,
        num_tables=num_tables,
        tracker=tracker,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        num_epochs=config["num_epochs_stage1"],
        learning_rate=config["learning_rate"],
        device=config["device"],
        early_stopping_patience=config["early_stopping_patience"],
        resume_from=config["resume_stage1"],
    )

    # Calibrate Stage 1
    logger.info("\nCalibrating Stage 1 model...")
    try:
        stage1_temperature = calibrate_model_stage1(
            stage1_model, stage1_cal_loader, config["device"], logger
        )
    except Exception as e:
        logger.error(f"Calibration failed for stage1: {e}")
        stage1_temperature = 1.0

    # Evaluate Stage 1
    stage1_results = evaluate_stage1(
        stage1_model,
        stage1_val_loader,
        stage1_val_dataset,
        config["device"],
        logger,
        tracker,
        epoch=config["num_epochs_stage1"],
    )

    # Save Stage 1 final artifacts
    stage1_final_path = run_dir / "stage1_complete.pt"
    torch.save(
        {
            "model_state_dict": stage1_model.state_dict(),
            "temperature": stage1_temperature,
            "table_to_idx": stage1_train_dataset.table_to_idx,
            "idx_to_table": stage1_train_dataset.idx_to_table,
            "config": config,
            "results": stage1_results,
        },
        stage1_final_path,
    )
    logger.info(f"Saved Stage 1 complete model to: {stage1_final_path}")
    tracker.log_artifact(str(stage1_final_path), artifact_type="stage1_model")

    # ========== STAGE 2 ==========
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: COLUMN MAPPING WITHIN TABLE")
    logger.info("=" * 80)

    # Reset checkpoint manager for stage 2
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir), max_checkpoints=config["max_checkpoints"]
    )

    # Load Stage 2 data
    logger.info("Loading Stage 2 datasets...")
    stage2_train_df = pd.read_csv(f"{config['data_dir']}/stage2_train.csv")
    stage2_val_df = pd.read_csv(f"{config['data_dir']}/stage2_val.csv")
    stage2_cal_df = pd.read_csv(f"{config['data_dir']}/stage2_calibration.csv")

    logger.info(f"Stage 2 Train samples: {len(stage2_train_df)}")
    logger.info(f"Stage 2 Val samples: {len(stage2_val_df)}")
    logger.info(f"Stage 2 Cal samples: {len(stage2_cal_df)}")

    # Create datasets
    stage2_train_dataset = ColumnMappingDataset(stage2_train_df, tokenizer)
    stage2_val_dataset = ColumnMappingDataset(stage2_val_df, tokenizer)
    stage2_cal_dataset = ColumnMappingDataset(stage2_cal_df, tokenizer)

    # Create dataloaders
    stage2_train_loader = DataLoader(
        stage2_train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    stage2_val_loader = DataLoader(
        stage2_val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    stage2_cal_loader = DataLoader(
        stage2_cal_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # Train Stage 2
    stage2_model, stage2_history = train_stage2(
        stage2_train_loader,
        stage2_val_loader,
        tracker=tracker,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        num_epochs=config["num_epochs_stage2"],
        learning_rate=config["learning_rate"],
        device=config["device"],
        early_stopping_patience=config["early_stopping_patience"],
        resume_from=config["resume_stage2"],
    )

    # Calibrate Stage 2
    logger.info("\nCalibrating Stage 2 model...")
    try:
        stage2_temperature = calibrate_model_stage2(
            stage2_model, stage2_cal_loader, config["device"], logger
        )
    except Exception as e:
        logger.error(f"Calibration failed for stage2: {e}")
        stage2_temperature = 1.0

    # Evaluate Stage 2
    stage2_results = evaluate_stage2(
        stage2_model,
        stage2_val_loader,
        config["device"],
        logger,
        tracker,
        epoch=config["num_epochs_stage2"],
    )

    # Save Stage 2 final artifacts
    stage2_final_path = run_dir / "stage2_complete.pt"
    torch.save(
        {
            "model_state_dict": stage2_model.state_dict(),
            "temperature": stage2_temperature,
            "config": config,
            "results": stage2_results,
        },
        stage2_final_path,
    )
    logger.info(f"Saved Stage 2 complete model to: {stage2_final_path}")
    tracker.log_artifact(str(stage2_final_path), artifact_type="stage2_model")

    # Save combined training history
    tracker.save_training_history({"stage1": stage1_history, "stage2": stage2_history})

    # Generate training plots
    plot_training_curves(stage1_history, stage2_history, run_dir, logger)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info("\nStage 1 Results:")
    logger.info(f"  - Top-1 Accuracy: {stage1_results.get('accuracy', 0.0):.4f}")
    logger.info(f"  - Top-3 Accuracy: {stage1_results.get('top3_accuracy', 0.0):.4f}")

    logger.info("\nStage 2 Results:")
    logger.info(f"  - Accuracy: {stage2_results.get('accuracy', 0.0):.4f}")
    logger.info(f"  - Precision: {stage2_results.get('precision', 0.0):.4f}")
    logger.info(f"  - Recall: {stage2_results.get('recall', 0.0):.4f}")
    logger.info(f"  - F1 Score: {stage2_results.get('f1', 0.0):.4f}")

    logger.info("\nSaved artifacts:")
    logger.info(f"  - {stage1_final_path}")
    logger.info(f"  - {stage2_final_path}")
    logger.info(f"  - {run_dir / 'training_history.json'}")
    logger.info(f"  - {run_dir / 'metrics.csv'}")
    logger.info(f"  - {checkpoint_dir}")
    logger.info("=" * 80)

    # Finalize experiment tracking
    tracker.finish()


def plot_training_curves(stage1_history, stage2_history, run_dir, logger):
    """Generate and save training curve plots"""
    logger.info("\nGenerating training curves...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Stage 1 Loss
    ax = axes[0, 0]
    ax.plot(stage1_history["train_loss"], label="Train Loss", marker="o")
    ax.plot(stage1_history["val_loss"], label="Val Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Stage 1: Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Stage 1 Accuracy
    ax = axes[0, 1]
    ax.plot(stage1_history["val_acc"], label="Top-1 Accuracy", marker="o")
    ax.plot(stage1_history["val_top3_acc"], label="Top-3 Accuracy", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Stage 1: Accuracy Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Stage 2 Loss
    ax = axes[1, 0]
    ax.plot(stage2_history["train_loss"], label="Train Loss", marker="o")
    ax.plot(stage2_history["val_loss"], label="Val Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Stage 2: Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Stage 2 Metrics
    ax = axes[1, 1]
    ax.plot(stage2_history["val_precision"], label="Precision", marker="o")
    ax.plot(stage2_history["val_recall"], label="Recall", marker="s")
    ax.plot(stage2_history["val_f1"], label="F1 Score", marker="^")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Stage 2: Performance Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = run_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved training curves to: {plot_path}")


if __name__ == "__main__":
    main()
