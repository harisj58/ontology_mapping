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
from collections import defaultdict
import matplotlib.pyplot as plt

# seaborn imported but not required by script logic; you can use it for plotting later
import seaborn as sns
import random


# ----------------- Reproducibility helper -----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        # infer encoder_dim from model config when possible
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
        # Encode file name
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token embedding if present
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Classify
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
        # Format: "[TABLE] standard_column"
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

        # Shared encoder
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.encoder_dim = getattr(self.encoder.config, "hidden_size", 384)

        # Similarity feature calculator
        self.similarity_features = nn.Sequential(
            nn.Linear(self.encoder_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        raw_input_ids,
        raw_attention_mask,
        standard_input_ids,
        standard_attention_mask,
    ):
        # Encode both inputs
        raw_embedding = self.encode_text(raw_input_ids, raw_attention_mask)
        standard_embedding = self.encode_text(
            standard_input_ids, standard_attention_mask
        )

        # Calculate similarity features
        similarity_vector = self.compute_similarity_features(
            raw_embedding, standard_embedding
        )

        # Get final prediction
        features = self.similarity_features(similarity_vector)
        logits = self.classifier(features)
        return torch.sigmoid(logits)

    def encode_text(self, input_ids, attention_mask):
        """Shared encoding function"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

    def compute_similarity_features(self, emb1, emb2):
        """Compute rich similarity features"""
        return torch.cat([emb1, emb2, torch.abs(emb1 - emb2), emb1 * emb2], dim=1)


# ============== TRAINING UTILITIES ==============


class TemperatureScaling(nn.Module):
    """Temperature scaling for confidence calibration"""

    def __init__(self):
        super().__init__()
        # initialize temperature > 0
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, logits):
        return logits / self.temperature


def calibrate_model_stage1(model, calibration_loader, device):
    """Calibrate Stage 1 model using temperature scaling"""
    temperature_model = TemperatureScaling().to(device)
    optimizer = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)

    model.eval()

    # Collect all logits and labels
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

    # Optimize temperature
    def eval_loss():
        optimizer.zero_grad()
        scaled = temperature_model(all_logits)
        loss = nn.CrossEntropyLoss()(scaled, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    t = float(temperature_model.temperature.item())
    print(f"Optimal temperature (stage1): {t:.4f}")
    return t


def calibrate_model_stage2(model, calibration_loader, device):
    """Calibrate Stage 2 model using temperature scaling"""
    temperature_model = TemperatureScaling().to(device)
    optimizer = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)

    model.eval()

    # Collect all logits and labels
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

    # Convert to logits for BCE (inverse sigmoid)
    all_logits = torch.logit(all_logits.clamp(1e-7, 1 - 7e-8))

    # Optimize temperature
    def eval_loss():
        optimizer.zero_grad()
        calibrated = torch.sigmoid(temperature_model(all_logits))
        loss = nn.BCELoss()(calibrated, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    t = float(temperature_model.temperature.item())
    print(f"Optimal temperature (stage2): {t:.4f}")
    return t


# ============== STAGE 1 TRAINING ==============


def train_stage1(
    train_loader,
    val_loader,
    num_tables,
    num_epochs=10,
    learning_rate=2e-5,
    device="cuda",
    save_dir="./models",
):
    """Train Stage 1 model: File name → Table classification"""

    print("=" * 80)
    print("TRAINING STAGE 1: File Name → Table Classification")
    print("=" * 80)

    model = FileToTableClassifier(num_tables=num_tables).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_top3_acc": [],
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / max(1, len(train_loader))

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

        # Calculate metrics
        val_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0

        # Top-3 accuracy (safe for < 3 classes)
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

        training_history["train_loss"].append(avg_train_loss)
        training_history["val_loss"].append(avg_val_loss)
        training_history["val_acc"].append(val_acc)
        training_history["val_top3_acc"].append(top3_acc)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val Top-3 Accuracy: {top3_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/stage1_best.pt")
            print(f"✓ Saved best model (accuracy: {best_val_acc:.4f})")

        print("-" * 80)

    # Attempt to load best model if exists
    best_path = f"{save_dir}/stage1_best.pt"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print(
            f"Warning: best model file not found at {best_path}. Returning last epoch model."
        )

    return model, training_history


# ============== STAGE 2 TRAINING ==============


def train_stage2(
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=2e-5,
    device="cuda",
    save_dir="./models",
):
    """Train Stage 2 model: Column mapping within table"""

    print("=" * 80)
    print("TRAINING STAGE 2: Column Mapping Within Table")
    print("=" * 80)

    model = SiameseColumnMapper().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_f1 = 0.0
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
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
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / max(1, len(train_loader))

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

        training_history["train_loss"].append(avg_train_loss)
        training_history["val_loss"].append(avg_val_loss)
        training_history["val_precision"].append(precision)
        training_history["val_recall"].append(recall)
        training_history["val_f1"].append(f1)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Precision: {precision:.4f}")
        print(f"Val Recall: {recall:.4f}")
        print(f"Val F1: {f1:.4f}")

        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/stage2_best.pt")
            print(f"✓ Saved best model (F1: {best_val_f1:.4f})")

        print("-" * 80)

    # Attempt to load best model if exists
    best_path = f"{save_dir}/stage2_best.pt"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print(
            f"Warning: best model file not found at {best_path}. Returning last epoch model."
        )

    return model, training_history


# ============== EVALUATION ==============


def evaluate_stage1(model, test_loader, dataset, device="cuda"):
    """Comprehensive evaluation of Stage 1 model"""
    print("\n" + "=" * 80)
    print("EVALUATING STAGE 1 MODEL")
    print("=" * 80)

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

    print(f"\nTop-1 Accuracy: {accuracy:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")

    # Confusion matrix
    cm = (
        confusion_matrix(all_labels, all_preds)
        if len(all_labels) > 0
        else np.array([[]])
    )

    # Per-class accuracy
    print("\nPer-Table Accuracy:")
    if cm.size:
        for i, table in enumerate(dataset.tables):
            denom = cm[i].sum() if i < cm.shape[0] else 0
            table_acc = cm[i, i] / denom if denom > 0 else 0
            print(f"  {table}: {table_acc:.4f}")
    else:
        print("  No samples in test set.")

    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_acc,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def evaluate_stage2(model, test_loader, device="cuda"):
    """Comprehensive evaluation of Stage 2 model"""
    print("\n" + "=" * 80)
    print("EVALUATING STAGE 2 MODEL")
    print("=" * 80)

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
            probs = outputs.squeeze().cpu().numpy()
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

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion matrix
    cm = (
        confusion_matrix(all_labels, all_preds)
        if len(all_labels) > 0
        else np.array([[]])
    )
    print(f"\nConfusion Matrix:")
    print(cm)

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
        "save_dir": "./models",
        "batch_size": 32,
        "num_epochs_stage1": 10,
        "num_epochs_stage2": 10,
        "learning_rate": 2e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_workers": 4,
    }

    print("Configuration:")
    print(json.dumps(config, indent=2))
    print(f"\nUsing device: {config['device']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_model"])

    # ========== STAGE 1 ==========
    print("\n" + "=" * 80)
    print("STAGE 1: FILE NAME → TABLE CLASSIFICATION")
    print("=" * 80)

    # Load Stage 1 data
    stage1_train_df = pd.read_csv(f"{config['data_dir']}/stage1_train.csv")
    stage1_val_df = pd.read_csv(f"{config['data_dir']}/stage1_val.csv")
    stage1_cal_df = pd.read_csv(f"{config['data_dir']}/stage1_calibration.csv")

    # Create datasets
    stage1_train_dataset = FileToTableDataset(stage1_train_df, tokenizer)
    stage1_val_dataset = FileToTableDataset(stage1_val_df, tokenizer)
    stage1_cal_dataset = FileToTableDataset(stage1_cal_df, tokenizer)

    num_tables = len(stage1_train_dataset.tables)
    print(f"Number of tables: {num_tables}")
    print(f"Tables: {stage1_train_dataset.tables}")

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
        num_epochs=config["num_epochs_stage1"],
        learning_rate=config["learning_rate"],
        device=config["device"],
        save_dir=config["save_dir"],
    )

    # Calibrate Stage 1
    print("\nCalibrating Stage 1 model...")
    try:
        stage1_temperature = calibrate_model_stage1(
            stage1_model, stage1_cal_loader, config["device"]
        )
    except Exception as e:
        print(f"Calibration failed for stage1: {e}")
        stage1_temperature = 1.0

    # Evaluate Stage 1
    stage1_results = evaluate_stage1(
        stage1_model, stage1_val_loader, stage1_val_dataset, config["device"]
    )

    # Save Stage 1 artifacts
    os.makedirs(config["save_dir"], exist_ok=True)
    torch.save(
        {
            "model_state_dict": stage1_model.state_dict(),
            "temperature": stage1_temperature,
            "table_to_idx": stage1_train_dataset.table_to_idx,
            "idx_to_table": stage1_train_dataset.idx_to_table,
            "config": config,
        },
        f"{config['save_dir']}/stage1_complete.pt",
    )

    # ========== STAGE 2 ==========
    print("\n" + "=" * 80)
    print("STAGE 2: COLUMN MAPPING WITHIN TABLE")
    print("=" * 80)

    # Load Stage 2 data
    stage2_train_df = pd.read_csv(f"{config['data_dir']}/stage2_train.csv")
    stage2_val_df = pd.read_csv(f"{config['data_dir']}/stage2_val.csv")
    stage2_cal_df = pd.read_csv(f"{config['data_dir']}/stage2_calibration.csv")

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
        num_epochs=config["num_epochs_stage2"],
        learning_rate=config["learning_rate"],
        device=config["device"],
        save_dir=config["save_dir"],
    )

    # Calibrate Stage 2
    print("\nCalibrating Stage 2 model...")
    try:
        stage2_temperature = calibrate_model_stage2(
            stage2_model, stage2_cal_loader, config["device"]
        )
    except Exception as e:
        print(f"Calibration failed for stage2: {e}")
        stage2_temperature = 1.0

    # Evaluate Stage 2
    stage2_results = evaluate_stage2(stage2_model, stage2_val_loader, config["device"])

    # Save Stage 2 artifacts
    torch.save(
        {
            "model_state_dict": stage2_model.state_dict(),
            "temperature": stage2_temperature,
            "config": config,
        },
        f"{config['save_dir']}/stage2_complete.pt",
    )

    # Save training history
    with open(f"{config['save_dir']}/training_history.json", "w") as f:
        json.dump({"stage1": stage1_history, "stage2": stage2_history}, f, indent=2)

    # Final summary prints
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Models saved to: {config['save_dir']}")
    print("\nStage 1 Results:")
    print(f"  - Top-1 Accuracy: {stage1_results.get('accuracy', 0.0):.4f}")
    print(f"  - Top-3 Accuracy: {stage1_results.get('top3_accuracy', 0.0):.4f}")

    print("\nStage 2 Results:")
    print(f"  - Accuracy: {stage2_results.get('accuracy', 0.0):.4f}")
    print(f"  - Precision: {stage2_results.get('precision', 0.0):.4f}")
    print(f"  - Recall: {stage2_results.get('recall', 0.0):.4f}")
    print(f"  - F1 Score: {stage2_results.get('f1', 0.0):.4f}")
    print("\nSaved artifacts:")
    print(f"  - {config['save_dir']}/stage1_complete.pt")
    print(f"  - {config['save_dir']}/stage2_complete.pt")
    print(f"  - {config['save_dir']}/training_history.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
