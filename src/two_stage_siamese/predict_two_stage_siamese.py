#!/usr/bin/env python3
"""
Interactive REPL for Two-Stage Column Mapping Inference
Provides top-3 table predictions, and for each table, top-3 column matches
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import os
from typing import List, Tuple, Dict
import numpy as np


# ============== MODEL DEFINITIONS (same as training script) ==============


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


# ============== INFERENCE ENGINE ==============


class TwoStageInferenceEngine:
    """Inference engine for two-stage column mapping"""

    def __init__(
        self,
        model_dir: str = "./models",
        schema_path: str = "./schema.json",
        device: str = None,
    ):
        """
        Initialize inference engine

        Args:
            model_dir: Directory containing trained models
            schema_path: Path to schema JSON file with standard columns per table
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir

        print(f"Loading models on device: {self.device}")

        # Load Stage 1 model
        stage1_checkpoint = torch.load(
            f"{model_dir}/stage1_complete.pt", map_location=self.device
        )

        self.stage1_config = stage1_checkpoint["config"]
        self.table_to_idx = stage1_checkpoint["table_to_idx"]
        self.idx_to_table = stage1_checkpoint["idx_to_table"]
        self.stage1_temperature = stage1_checkpoint.get("temperature", 1.0)

        num_tables = len(self.table_to_idx)
        self.stage1_model = FileToTableClassifier(
            encoder_model=self.stage1_config["encoder_model"], num_tables=num_tables
        ).to(self.device)
        self.stage1_model.load_state_dict(stage1_checkpoint["model_state_dict"])
        self.stage1_model.eval()

        # Load Stage 2 model
        stage2_checkpoint = torch.load(
            f"{model_dir}/stage2_complete.pt", map_location=self.device
        )

        self.stage2_temperature = stage2_checkpoint.get("temperature", 1.0)

        self.stage2_model = SiameseColumnMapper(
            encoder_model=self.stage1_config["encoder_model"]
        ).to(self.device)
        self.stage2_model.load_state_dict(stage2_checkpoint["model_state_dict"])
        self.stage2_model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.stage1_config["encoder_model"]
        )

        # Load schema
        self.schema = self._load_schema(schema_path)

        print(f"✓ Loaded Stage 1 model (temperature: {self.stage1_temperature:.4f})")
        print(f"✓ Loaded Stage 2 model (temperature: {self.stage2_temperature:.4f})")
        print(f"✓ Available tables: {list(self.table_to_idx.keys())}")

    def _load_schema(self, schema_path: str) -> Dict[str, List[str]]:
        """Load schema from JSON file"""
        if not os.path.exists(schema_path):
            print(f"Warning: Schema file not found at {schema_path}")
            print("Using default schema from training data...")
            # Return empty dict, will need to be populated
            return {}

        with open(schema_path, "r") as f:
            raw_schema = json.load(f)

        # Extract standard_columns from the nested structure
        schema = {}
        for table_name, table_config in raw_schema.items():
            if isinstance(table_config, dict) and "standard_columns" in table_config:
                schema[table_name] = table_config["standard_columns"]
            elif isinstance(table_config, list):
                # Fallback for simple list format
                schema[table_name] = table_config

        print(f"✓ Loaded schema with {len(schema)} tables")
        for table_name, columns in schema.items():
            print(f"  - {table_name}: {len(columns)} columns")

        return schema

    def predict_table(self, file_name: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict top-k tables for a file name

        Args:
            file_name: Input file name
            top_k: Number of top predictions to return

        Returns:
            List of (table_name, confidence_score) tuples
        """
        # Tokenize file name
        encoding = self.tokenizer(
            file_name,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get predictions
        with torch.no_grad():
            logits = self.stage1_model(input_ids, attention_mask)
            # Apply temperature scaling
            logits = logits / self.stage1_temperature
            probs = torch.softmax(logits, dim=1).squeeze()

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

        results = [
            (self.idx_to_table[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]

        return results

    def predict_columns(
        self, raw_column: str, table_name: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Predict top-k standard columns for a raw column within a table context

        Args:
            raw_column: Input raw column name
            table_name: Table context
            top_k: Number of top predictions to return

        Returns:
            List of (standard_column, confidence_score) tuples
        """
        if table_name not in self.schema:
            print(f"Warning: Table '{table_name}' not found in schema")
            return []

        standard_columns = self.schema[table_name]

        # Tokenize raw column
        raw_encoding = self.tokenizer(
            raw_column,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        raw_input_ids = raw_encoding["input_ids"].to(self.device)
        raw_attention_mask = raw_encoding["attention_mask"].to(self.device)

        # Score each standard column
        scores = []

        with torch.no_grad():
            for std_col in standard_columns:
                # Format with table context
                standard_with_context = f"[{table_name}] {std_col}"

                std_encoding = self.tokenizer(
                    standard_with_context,
                    max_length=64,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                std_input_ids = std_encoding["input_ids"].to(self.device)
                std_attention_mask = std_encoding["attention_mask"].to(self.device)

                # Get similarity score
                output = self.stage2_model(
                    raw_input_ids,
                    raw_attention_mask,
                    std_input_ids,
                    std_attention_mask,
                )

                score = output.squeeze().item()
                scores.append((std_col, score))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def predict(
        self, file_name: str, raw_column: str, top_tables: int = 3, top_columns: int = 3
    ) -> List[Dict]:
        """
        Full two-stage prediction

        Args:
            file_name: Input file name
            raw_column: Input raw column name
            top_tables: Number of top table predictions
            top_columns: Number of top column predictions per table

        Returns:
            List of prediction dictionaries
        """
        # Stage 1: Predict tables
        table_predictions = self.predict_table(file_name, top_k=top_tables)

        # Stage 2: For each table, predict columns
        results = []

        for table_name, table_confidence in table_predictions:
            column_predictions = self.predict_columns(
                raw_column, table_name, top_k=top_columns
            )

            for std_column, column_confidence in column_predictions:
                results.append(
                    {
                        "input_file_name": file_name,
                        "input_column_name": raw_column,
                        "output_table_name": table_name,
                        "output_column_name": std_column,
                        "table_confidence": table_confidence,
                        "column_confidence": column_confidence,
                        "combined_confidence": table_confidence * column_confidence,
                    }
                )

        return results


# ============== REPL INTERFACE ==============


def print_results(results: List[Dict]):
    """Pretty print results in table format"""
    if not results:
        print("\nNo results found.")
        return

    print("\n" + "=" * 140)
    print(
        f"{'Input File Name':<30} | {'Input Column Name':<25} | {'Output Table Name':<20} | {'Output Column Name':<25} | {'Confidence Score':<10}"
    )
    print("=" * 140)

    for result in results:
        print(
            f"{result['input_file_name']:<30} | "
            f"{result['input_column_name']:<25} | "
            f"{result['output_table_name']:<20} | "
            f"{result['output_column_name']:<25} | "
            f"{result['combined_confidence']:.4f}"
        )

    print("=" * 140)
    print(f"\nShowing {len(results)} predictions")
    print(
        f"(Top {len(set(r['output_table_name'] for r in results))} tables × Top {len([r for r in results if r['output_table_name'] == results[0]['output_table_name']])} columns per table)\n"
    )


def run_repl():
    """Run interactive REPL"""
    print("=" * 80)
    print("TWO-STAGE COLUMN MAPPING INFERENCE REPL")
    print("=" * 80)
    print("\nInitializing inference engine...")

    # Initialize engine
    try:
        engine = TwoStageInferenceEngine(
            model_dir="./models",
            schema_path="./two_stage_datasets/domain_metadata.json",
        )
    except Exception as e:
        print(f"\nError loading models: {e}")
        print("\nMake sure you have:")
        print("  1. Trained models in ./models/ directory")
        print("  2. schema.json file with standard columns per table")
        return

    print("\n" + "=" * 80)
    print("Ready for inference!")
    print("Commands:")
    print("  - Enter input in format: <file_name> | <column_name>")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'help' for more information")
    print("=" * 80)

    while True:
        try:
            # Get input
            user_input = input("\n>>> ").strip()

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nExiting REPL. Goodbye!")
                break

            # Check for help
            if user_input.lower() == "help":
                print("\nUsage:")
                print("  Enter: <file_name> | <column_name>")
                print("\nExample:")
                print("  >>> customer_data.csv | cust_id")
                print("\nThe system will return:")
                print("  - Top 3 table predictions")
                print("  - For each table, top 3 column matches")
                print("  - Combined confidence scores")
                continue

            # Parse input
            if "|" not in user_input:
                print("Error: Please use format: <file_name> | <column_name>")
                continue

            parts = user_input.split("|")
            if len(parts) != 2:
                print("Error: Please use format: <file_name> | <column_name>")
                continue

            file_name = parts[0].strip()
            column_name = parts[1].strip()

            if not file_name or not column_name:
                print("Error: Both file name and column name are required")
                continue

            # Run inference
            print(f"\nPredicting for:")
            print(f"  File: {file_name}")
            print(f"  Column: {column_name}")

            results = engine.predict(
                file_name=file_name, raw_column=column_name, top_tables=3, top_columns=3
            )

            # Display results
            print_results(results)

        except KeyboardInterrupt:
            print("\n\nExiting REPL. Goodbye!")
            break
        except Exception as e:
            print(f"\nError during inference: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    run_repl()
