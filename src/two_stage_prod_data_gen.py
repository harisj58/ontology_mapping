import pandas as pd
import numpy as np
import random
import string
import json
from collections import defaultdict
from faker import Faker
from sklearn.model_selection import train_test_split
import os


class TwoStageDatasetGenerator:
    """Generate datasets for two-stage hierarchical column mapping"""

    def __init__(self, random_seed=42):
        """
        Initialize the dataset generator

        Args:
            random_seed (int): Random seed for reproducibility
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.fake = Faker()
        Faker.seed(random_seed)

        # Define business domains
        self.domains = self._define_domains()

        # File naming patterns
        self.file_patterns = self._define_file_patterns()

        # Column transformation rules
        self.transformation_rules = self._define_transformation_rules()

    def _define_domains(self):
        """Define comprehensive business domains"""
        return {
            "customers": {
                "standard_columns": [
                    "customer_id",
                    "customer_name",
                    "customer_email",
                    "customer_phone",
                    "customer_address",
                    "customer_city",
                    "customer_state",
                    "customer_zip",
                    "customer_country",
                    "customer_type",
                    "registration_date",
                    "last_login_date",
                    "customer_status",
                    "preferred_language",
                    "customer_segment",
                    "loyalty_points",
                    "date_of_birth",
                    "gender",
                    "occupation",
                    "income_range",
                ],
                "semantic_groups": {
                    "identifiers": ["customer_id"],
                    "personal_info": [
                        "customer_name",
                        "date_of_birth",
                        "gender",
                        "occupation",
                    ],
                    "contact": ["customer_email", "customer_phone", "customer_address"],
                    "location": [
                        "customer_city",
                        "customer_state",
                        "customer_zip",
                        "customer_country",
                    ],
                },
            },
            "orders": {
                "standard_columns": [
                    "order_id",
                    "customer_id",
                    "order_date",
                    "order_status",
                    "order_total",
                    "payment_method",
                    "shipping_address",
                    "shipping_city",
                    "shipping_state",
                    "shipping_zip",
                    "delivery_date",
                    "order_notes",
                    "discount_amount",
                    "tax_amount",
                    "shipping_cost",
                    "order_source",
                    "coupon_code",
                    "estimated_delivery",
                    "actual_delivery",
                    "order_priority",
                ],
                "semantic_groups": {
                    "identifiers": ["order_id", "customer_id"],
                    "financial": [
                        "order_total",
                        "discount_amount",
                        "tax_amount",
                        "shipping_cost",
                    ],
                    "temporal": [
                        "order_date",
                        "delivery_date",
                        "estimated_delivery",
                        "actual_delivery",
                    ],
                },
            },
            "products": {
                "standard_columns": [
                    "product_id",
                    "product_name",
                    "product_description",
                    "product_category",
                    "product_subcategory",
                    "product_brand",
                    "product_price",
                    "product_cost",
                    "product_sku",
                    "product_barcode",
                    "stock_quantity",
                    "reorder_level",
                    "supplier_id",
                    "product_weight",
                    "product_dimensions",
                    "product_color",
                    "product_size",
                    "product_status",
                    "date_added",
                    "last_updated",
                ],
                "semantic_groups": {
                    "identifiers": ["product_id", "product_sku", "product_barcode"],
                    "descriptive": [
                        "product_name",
                        "product_description",
                        "product_category",
                    ],
                    "financial": ["product_price", "product_cost"],
                },
            },
            "employees": {
                "standard_columns": [
                    "employee_id",
                    "employee_name",
                    "employee_email",
                    "employee_phone",
                    "employee_department",
                    "employee_position",
                    "employee_salary",
                    "hire_date",
                    "termination_date",
                    "employee_status",
                    "manager_id",
                    "employee_address",
                    "employee_city",
                    "employee_state",
                    "employee_zip",
                    "date_of_birth",
                    "emergency_contact",
                    "emergency_phone",
                    "performance_rating",
                    "years_of_service",
                ],
                "semantic_groups": {
                    "identifiers": ["employee_id"],
                    "personal": ["employee_name", "date_of_birth"],
                    "contact": ["employee_email", "employee_phone"],
                },
            },
            "transactions": {
                "standard_columns": [
                    "transaction_id",
                    "order_id",
                    "customer_id",
                    "transaction_date",
                    "transaction_amount",
                    "transaction_type",
                    "transaction_status",
                    "payment_method",
                    "payment_processor",
                    "transaction_fee",
                    "currency_code",
                    "exchange_rate",
                    "card_last_four",
                    "authorization_code",
                    "settlement_date",
                    "refund_amount",
                    "refund_date",
                    "transaction_notes",
                    "merchant_id",
                    "terminal_id",
                ],
                "semantic_groups": {
                    "identifiers": ["transaction_id", "order_id", "customer_id"],
                    "financial": [
                        "transaction_amount",
                        "transaction_fee",
                        "refund_amount",
                    ],
                    "temporal": ["transaction_date", "settlement_date", "refund_date"],
                },
            },
        }

    def _define_file_patterns(self):
        """Define file naming patterns for each table"""
        return {
            "customers": {
                "explicit": [
                    "customers.{ext}",
                    "customer_data.{ext}",
                    "customer_export.{ext}",
                    "customers_{date}.{ext}",
                    "customer_list_{version}.{ext}",
                    "raw_customers.{ext}",
                    "staging_customers.{ext}",
                    "customer_master.{ext}",
                ],
                "implicit": [
                    "clients.{ext}",
                    "client_database.{ext}",
                    "user_data.{ext}",
                    "crm_export.{ext}",
                    "member_list.{ext}",
                    "accounts.{ext}",
                    "contact_list.{ext}",
                    "subscriber_data.{ext}",
                ],
                "ambiguous": [
                    "data.{ext}",
                    "export_{date}.{ext}",
                    "report.{ext}",
                    "database_dump.{ext}",
                    "master_file.{ext}",
                ],
            },
            "orders": {
                "explicit": [
                    "orders.{ext}",
                    "order_data.{ext}",
                    "order_history.{ext}",
                    "orders_{date}.{ext}",
                    "order_export_{version}.{ext}",
                    "sales_orders.{ext}",
                    "order_details.{ext}",
                ],
                "implicit": [
                    "sales.{ext}",
                    "purchases.{ext}",
                    "sales_data.{ext}",
                    "transactions_report.{ext}",
                    "invoice_data.{ext}",
                    "purchase_history.{ext}",
                    "bookings.{ext}",
                ],
                "ambiguous": [
                    "data.{ext}",
                    "export_{date}.{ext}",
                    "report.{ext}",
                    "sales_report.{ext}",
                ],
            },
            "products": {
                "explicit": [
                    "products.{ext}",
                    "product_catalog.{ext}",
                    "product_master.{ext}",
                    "products_{date}.{ext}",
                    "product_list.{ext}",
                    "product_data.{ext}",
                ],
                "implicit": [
                    "inventory.{ext}",
                    "catalog.{ext}",
                    "items.{ext}",
                    "sku_list.{ext}",
                    "merchandise.{ext}",
                    "stock_data.{ext}",
                    "item_master.{ext}",
                ],
                "ambiguous": [
                    "data.{ext}",
                    "master.{ext}",
                    "catalog_{date}.{ext}",
                ],
            },
            "employees": {
                "explicit": [
                    "employees.{ext}",
                    "employee_data.{ext}",
                    "employee_roster.{ext}",
                    "employees_{date}.{ext}",
                    "staff_list.{ext}",
                    "employee_master.{ext}",
                ],
                "implicit": [
                    "staff.{ext}",
                    "workforce.{ext}",
                    "personnel.{ext}",
                    "hr_data.{ext}",
                    "payroll_data.{ext}",
                    "team_roster.{ext}",
                ],
                "ambiguous": [
                    "data.{ext}",
                    "roster.{ext}",
                    "hr_export.{ext}",
                ],
            },
            "transactions": {
                "explicit": [
                    "transactions.{ext}",
                    "transaction_data.{ext}",
                    "transaction_log.{ext}",
                    "transactions_{date}.{ext}",
                    "payment_transactions.{ext}",
                ],
                "implicit": [
                    "payments.{ext}",
                    "payment_data.{ext}",
                    "payment_history.{ext}",
                    "financial_transactions.{ext}",
                    "billing_data.{ext}",
                ],
                "ambiguous": [
                    "data.{ext}",
                    "financial_report.{ext}",
                    "payment_log.{ext}",
                ],
            },
        }

    def _define_transformation_rules(self):
        """Define rules for transforming standard columns to raw columns"""
        return {
            "abbreviation": {
                "customer": ["cust", "c", "cstmr", "client", "clnt"],
                "order": ["ord", "o", "purchase", "sale"],
                "product": ["prod", "p", "item", "itm"],
                "employee": ["emp", "e", "staff", "stf"],
                "transaction": ["trans", "txn", "t", "payment", "pmt"],
                "address": ["addr", "add", "location", "loc"],
                "phone": ["ph", "tel", "telephone", "mobile", "mob"],
                "email": ["mail", "e_mail", "electronic_mail"],
                "date": ["dt", "d", "time", "timestamp", "ts"],
                "status": ["stat", "sts", "state"],
                "amount": ["amt", "total", "value", "val"],
                "number": ["num", "no", "nbr", "#"],
                "identifier": ["id", "key", "code", "ref"],
                "name": ["nm", "title", "label"],
                "description": ["desc", "details", "info"],
                "quantity": ["qty", "count", "cnt"],
                "price": ["prc", "cost", "rate"],
            },
            "synonyms": {
                "customer": [
                    "client",
                    "buyer",
                    "purchaser",
                    "account",
                    "member",
                    "user",
                ],
                "order": ["purchase", "sale", "transaction", "booking", "reservation"],
                "product": ["item", "merchandise", "goods", "sku", "article"],
                "employee": ["staff", "worker", "personnel", "team_member"],
                "email": ["electronic_mail", "e_mail", "mail_address"],
                "phone": ["telephone", "mobile", "contact_number"],
                "address": ["location", "residence", "street_address"],
                "date": ["timestamp", "datetime", "time"],
                "total": ["amount", "sum", "value", "grand_total"],
                "status": ["state", "condition", "stage"],
            },
            "prefixes": ["raw_", "src_", "orig_", "old_", "legacy_", "temp_", "new_"],
            "suffixes": [
                "_raw",
                "_src",
                "_orig",
                "_old",
                "_legacy",
                "_temp",
                "_new",
                "_v1",
                "_v2",
            ],
        }

    # ============== STAGE 1: FILE NAME TO TABLE ==============

    def generate_stage1_dataset(
        self, n_samples=50000, val_split=0.15, calibration_split=0.15
    ):
        """
        Generate Stage 1 dataset: File Name → Table Classification

        Args:
            n_samples: Total number of samples to generate
            val_split: Validation split ratio
            calibration_split: Calibration split ratio

        Returns:
            train_df, val_df, calibration_df
        """
        tables = list(self.domains.keys())
        samples_per_table = n_samples // len(tables)

        all_samples = []

        for table in tables:
            patterns = self.file_patterns[table]

            # Explicit file names (40%)
            explicit_samples = self._generate_explicit_filenames(
                table, patterns["explicit"], int(samples_per_table * 0.4)
            )

            # Implicit file names (30%)
            implicit_samples = self._generate_implicit_filenames(
                table, patterns["implicit"], int(samples_per_table * 0.3)
            )

            # Ambiguous file names (20%)
            ambiguous_samples = self._generate_ambiguous_filenames(
                table, patterns["ambiguous"], int(samples_per_table * 0.2)
            )

            # Noisy/misleading file names (10%)
            noisy_samples = self._generate_noisy_filenames(
                table, tables, int(samples_per_table * 0.1)
            )

            all_samples.extend(explicit_samples)
            all_samples.extend(implicit_samples)
            all_samples.extend(ambiguous_samples)
            all_samples.extend(noisy_samples)

        # Shuffle
        random.shuffle(all_samples)

        # Create DataFrame
        df = pd.DataFrame(all_samples)

        # Split into train, val, calibration
        train_val_df, calibration_df = train_test_split(
            df, test_size=calibration_split, stratify=df["table"], random_state=42
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_split / (1 - calibration_split),
            stratify=train_val_df["table"],
            random_state=42,
        )

        return train_df, val_df, calibration_df

    def _generate_explicit_filenames(self, table, patterns, n_samples):
        """Generate explicit file names that clearly indicate the table"""
        samples = []
        extensions = ["csv", "xlsx", "parquet", "tsv", "txt"]
        dates = ["2024", "2023", "jan_2024", "q1_2024", "2024_01", "20240115"]
        versions = ["v1", "v2", "final", "latest", "backup"]

        for _ in range(n_samples):
            pattern = random.choice(patterns)
            ext = random.choice(extensions)

            # Replace placeholders
            filename = pattern.format(
                ext=ext,
                date=random.choice(dates) if "{date}" in pattern else "",
                version=random.choice(versions) if "{version}" in pattern else "",
            )

            # Apply case variations
            filename = self._apply_case_variation(filename)

            samples.append(
                {
                    "file_name": filename,
                    "table": table,
                }
            )

        return samples

    def _generate_implicit_filenames(self, table, patterns, n_samples):
        """Generate implicit file names (domain-related but not direct)"""
        samples = []
        extensions = ["csv", "xlsx", "parquet", "tsv"]

        for _ in range(n_samples):
            pattern = random.choice(patterns)
            ext = random.choice(extensions)
            filename = pattern.format(ext=ext)

            # Add optional prefixes/suffixes
            if random.random() < 0.3:
                filename = (
                    random.choice(["raw_", "staging_", "prod_", "dev_"]) + filename
                )

            filename = self._apply_case_variation(filename)

            samples.append(
                {
                    "file_name": filename,
                    "table": table,
                }
            )

        return samples

    def _generate_ambiguous_filenames(self, table, patterns, n_samples):
        """Generate ambiguous file names that could belong to multiple tables"""
        samples = []
        extensions = ["csv", "xlsx", "txt", "dat"]
        dates = ["2024", "jan", "q1", "20240115"]

        for _ in range(n_samples):
            pattern = random.choice(patterns)
            ext = random.choice(extensions)
            filename = pattern.format(
                ext=ext, date=random.choice(dates) if "{date}" in pattern else ""
            )

            filename = self._apply_case_variation(filename)

            samples.append(
                {
                    "file_name": filename,
                    "table": table,
                }
            )

        return samples

    def _generate_noisy_filenames(self, table, all_tables, n_samples):
        """Generate noisy/misleading file names"""
        samples = []
        extensions = ["csv", "xlsx"]

        for _ in range(n_samples):
            # Pick 2 random table names and combine them
            other_tables = [t for t in all_tables if t != table]
            misleading_table = random.choice(other_tables)

            # Create misleading names
            patterns = [
                f"{misleading_table}_{table}.{{ext}}",
                f"{table}_{misleading_table}_report.{{ext}}",
                f"combined_{table}_{misleading_table}.{{ext}}",
                f"{misleading_table}_with_{table}.{{ext}}",
            ]

            pattern = random.choice(patterns)
            ext = random.choice(extensions)
            filename = pattern.format(ext=ext)

            samples.append(
                {
                    "file_name": filename,
                    "table": table,  # Ground truth is still the original table
                }
            )

        return samples

    def _apply_case_variation(self, filename):
        """Apply random case variations to filename"""
        variation = random.choice(["lower", "upper", "title", "mixed", "original"])

        if variation == "lower":
            return filename.lower()
        elif variation == "upper":
            return filename.upper()
        elif variation == "title":
            return filename.title()
        elif variation == "mixed":
            # Random mix of upper and lower
            return "".join(
                c.upper() if random.random() > 0.5 else c.lower() for c in filename
            )
        else:
            return filename

    # ============== STAGE 2: COLUMN MAPPING WITHIN TABLE ==============

    def generate_stage2_dataset(
        self, n_samples=150000, val_split=0.15, calibration_split=0.15
    ):
        """
        Generate Stage 2 dataset: Raw Column → Standard Column (within table context)

        Args:
            n_samples: Total number of samples to generate
            val_split: Validation split ratio
            calibration_split: Calibration split ratio

        Returns:
            train_df, val_df, calibration_df
        """
        all_samples = []

        # Calculate samples per table based on number of columns
        table_weights = {
            table: len(self.domains[table]["standard_columns"])
            for table in self.domains.keys()
        }
        total_columns = sum(table_weights.values())

        for table, num_columns in table_weights.items():
            table_samples = int(n_samples * (num_columns / total_columns))

            # Generate positive and negative examples
            positive_samples = self._generate_positive_mappings(
                table, table_samples // 4
            )
            negative_samples = self._generate_negative_mappings(
                table, table_samples * 3 // 4
            )

            all_samples.extend(positive_samples)
            all_samples.extend(negative_samples)

        # Shuffle
        random.shuffle(all_samples)

        # Create DataFrame
        df = pd.DataFrame(all_samples)

        # Split into train, val, calibration
        train_val_df, calibration_df = train_test_split(
            df,
            test_size=calibration_split,
            stratify=df["table_context"],
            random_state=42,
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_split / (1 - calibration_split),
            stratify=train_val_df["table_context"],
            random_state=42,
        )

        return train_df, val_df, calibration_df

    def _generate_positive_mappings(self, table, n_samples):
        """Generate positive (matching) column pairs"""
        samples = []
        standard_columns = self.domains[table]["standard_columns"]

        samples_per_column = max(1, n_samples // len(standard_columns))

        for standard_col in standard_columns:
            for _ in range(samples_per_column):
                raw_col = self._transform_column_name(standard_col)

                samples.append(
                    {
                        "raw_column": raw_col,
                        "standard_column": standard_col,
                        "table_context": table,
                        "label": 1,
                    }
                )

        return samples

    def _generate_negative_mappings(self, table, n_samples):
        """Generate negative (non-matching) column pairs"""
        samples = []
        standard_columns = self.domains[table]["standard_columns"]

        for _ in range(n_samples):
            # Pick a random standard column
            standard_col = random.choice(standard_columns)

            # Pick a different column from the same table (hard negative)
            other_col = random.choice(
                [c for c in standard_columns if c != standard_col]
            )

            # Transform it to create a misleading raw column
            raw_col = self._transform_column_name(other_col)

            samples.append(
                {
                    "raw_column": raw_col,
                    "standard_column": standard_col,
                    "table_context": table,
                    "label": 0,
                }
            )

        return samples

    def _transform_column_name(self, column_name):
        """Transform a standard column name into a raw column variation"""
        transformations = [
            self._apply_direct_match,
            self._apply_abbreviation,
            self._apply_synonym,
            self._apply_case_change,
            self._apply_prefix_suffix,
            self._apply_separator_change,
            self._apply_vowel_removal,
            self._apply_complex_transformation,
        ]

        # Apply one or more transformations
        transformed = column_name
        num_transforms = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]

        for _ in range(num_transforms):
            transform_func = random.choice(transformations)
            transformed = transform_func(transformed)

        return transformed

    def _apply_direct_match(self, column_name):
        """Return exact match or simple case variation"""
        variations = [
            column_name,
            column_name.upper(),
            column_name.title(),
            column_name.replace("_", ""),
        ]
        return random.choice(variations)

    def _apply_abbreviation(self, column_name):
        """Apply abbreviation rules"""
        parts = column_name.split("_")
        transformed_parts = []

        for part in parts:
            if part in self.transformation_rules["abbreviation"]:
                abbrev = random.choice(self.transformation_rules["abbreviation"][part])
                transformed_parts.append(abbrev)
            else:
                transformed_parts.append(part)

        return "_".join(transformed_parts)

    def _apply_synonym(self, column_name):
        """Apply synonym replacement"""
        parts = column_name.split("_")
        transformed_parts = []

        for part in parts:
            if part in self.transformation_rules["synonyms"]:
                synonym = random.choice(self.transformation_rules["synonyms"][part])
                transformed_parts.append(synonym)
            else:
                transformed_parts.append(part)

        return "_".join(transformed_parts)

    def _apply_case_change(self, column_name):
        """Apply various case changes"""
        cases = [
            column_name.lower(),
            column_name.upper(),
            column_name.title().replace("_", ""),
            "".join(word.capitalize() for word in column_name.split("_")),  # CamelCase
        ]
        return random.choice(cases)

    def _apply_prefix_suffix(self, column_name):
        """Add prefix or suffix"""
        if random.random() < 0.5:
            prefix = random.choice(self.transformation_rules["prefixes"])
            return prefix + column_name
        else:
            suffix = random.choice(self.transformation_rules["suffixes"])
            return column_name + suffix

    def _apply_separator_change(self, column_name):
        """Change separator style"""
        separators = ["_", "-", ".", ""]
        new_sep = random.choice(separators)
        return column_name.replace("_", new_sep)

    def _apply_vowel_removal(self, column_name):
        """Remove vowels (common abbreviation technique)"""
        vowels = "aeiou"
        parts = column_name.split("_")
        transformed_parts = []

        for part in parts:
            if len(part) > 3:  # Only remove vowels from longer words
                transformed = "".join(c for c in part if c.lower() not in vowels)
                transformed_parts.append(transformed if transformed else part)
            else:
                transformed_parts.append(part)

        return "_".join(transformed_parts)

    def _apply_complex_transformation(self, column_name):
        """Apply complex transformations (concatenation, truncation)"""
        transformations = [
            column_name.replace("_", ""),  # Remove all separators
            column_name[:10],  # Truncate
            column_name + "_field",  # Add generic suffix
            "col_" + column_name,  # Add generic prefix
            column_name.split("_")[0],  # Take first part only
        ]
        return random.choice(transformations)

    # ============== UTILITY METHODS ==============

    def save_datasets(self, output_dir="./datasets"):
        """Generate and save all datasets"""
        os.makedirs(output_dir, exist_ok=True)

        print("Generating Stage 1 datasets...")
        stage1_train, stage1_val, stage1_cal = self.generate_stage1_dataset()

        stage1_train.to_csv(f"{output_dir}/stage1_train.csv", index=False)
        stage1_val.to_csv(f"{output_dir}/stage1_val.csv", index=False)
        stage1_cal.to_csv(f"{output_dir}/stage1_calibration.csv", index=False)

        print(
            f"Stage 1 - Train: {len(stage1_train)}, Val: {len(stage1_val)}, Cal: {len(stage1_cal)}"
        )
        print(f"Stage 1 table distribution:\n{stage1_train['table'].value_counts()}")

        print("\nGenerating Stage 2 datasets...")
        stage2_train, stage2_val, stage2_cal = self.generate_stage2_dataset()

        stage2_train.to_csv(f"{output_dir}/stage2_train.csv", index=False)
        stage2_val.to_csv(f"{output_dir}/stage2_val.csv", index=False)
        stage2_cal.to_csv(f"{output_dir}/stage2_calibration.csv", index=False)

        print(
            f"Stage 2 - Train: {len(stage2_train)}, Val: {len(stage2_val)}, Cal: {len(stage2_cal)}"
        )
        print(
            f"Stage 2 table distribution:\n{stage2_train['table_context'].value_counts()}"
        )
        print(f"Stage 2 label distribution:\n{stage2_train['label'].value_counts()}")

        # Save domain metadata
        with open(f"{output_dir}/domain_metadata.json", "w") as f:
            json.dump(self.domains, f, indent=2)

        print(f"\nAll datasets saved to {output_dir}")

    def get_statistics(self):
        """Get statistics about the defined domains"""
        stats = {"num_tables": len(self.domains), "tables": {}}

        for table, config in self.domains.items():
            stats["tables"][table] = {
                "num_columns": len(config["standard_columns"]),
                "columns": config["standard_columns"],
                "num_semantic_groups": len(config["semantic_groups"]),
            }

        return stats


# Example usage
if __name__ == "__main__":
    generator = TwoStageDatasetGenerator(random_seed=42)

    # Print statistics
    stats = generator.get_statistics()
    print("Domain Statistics:")
    print(json.dumps(stats, indent=2))

    # Generate and save all datasets
    generator.save_datasets(output_dir="./two_stage_datasets")

    # Preview samples
    print("\n" + "=" * 80)
    print("STAGE 1 SAMPLE DATA:")
    print("=" * 80)
    stage1_train, _, _ = generator.generate_stage1_dataset(n_samples=1000)
    print(stage1_train.head(10))

    print("\n" + "=" * 80)
    print("STAGE 2 SAMPLE DATA:")
    print("=" * 80)
    stage2_train, _, _ = generator.generate_stage2_dataset(n_samples=1000)
    print(stage2_train.head(10))
    print(f"\nPositive samples: {(stage2_train['label'] == 1).sum()}")
    print(f"Negative samples: {(stage2_train['label'] == 0).sum()}")
