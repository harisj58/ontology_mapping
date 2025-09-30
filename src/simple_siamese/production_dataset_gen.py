import pandas as pd
import numpy as np
import random
import string
import json
from collections import defaultdict
from faker import Faker
from sklearn.model_selection import train_test_split
import os


class NeuralDatasetGenerator:
    """Generate large-scale synthetic dataset for neural network training"""

    def __init__(self, target_size=200000, random_seed=42):
        """
        Initialize the dataset generator

        Args:
            target_size (int): Target number of training examples
            random_seed (int): Random seed for reproducibility
        """
        self.target_size = target_size
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.fake = Faker()
        Faker.seed(random_seed)

        # Extended business domains for neural network training
        self.domains = self._define_extended_domains()
        self.linguistic_patterns = self._define_linguistic_patterns()
        self.augmentation_strategies = self._define_augmentation_strategies()

    def _define_extended_domains(self):
        """Define comprehensive business domains with rich vocabulary"""
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
                    "behavioral": [
                        "customer_type",
                        "customer_segment",
                        "loyalty_points",
                    ],
                    "temporal": ["registration_date", "last_login_date"],
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
                    "location": [
                        "shipping_address",
                        "shipping_city",
                        "shipping_state",
                        "shipping_zip",
                    ],
                    "temporal": [
                        "order_date",
                        "delivery_date",
                        "estimated_delivery",
                        "actual_delivery",
                    ],
                    "status": ["order_status", "order_priority"],
                    "metadata": [
                        "payment_method",
                        "order_source",
                        "coupon_code",
                        "order_notes",
                    ],
                },
            },
            "products": {
                "standard_columns": [
                    "product_id",
                    "product_name",
                    "product_category",
                    "product_price",
                    "product_cost",
                    "product_description",
                    "product_weight",
                    "product_dimensions",
                    "inventory_quantity",
                    "supplier_id",
                    "created_date",
                    "last_updated",
                    "brand_name",
                    "model_number",
                    "color",
                    "size",
                    "material",
                    "warranty_period",
                    "product_rating",
                    "review_count",
                ],
                "semantic_groups": {
                    "identifiers": ["product_id", "supplier_id", "model_number"],
                    "descriptive": [
                        "product_name",
                        "product_description",
                        "brand_name",
                    ],
                    "financial": ["product_price", "product_cost"],
                    "physical": [
                        "product_weight",
                        "product_dimensions",
                        "color",
                        "size",
                        "material",
                    ],
                    "inventory": ["inventory_quantity", "supplier_id"],
                    "temporal": ["created_date", "last_updated"],
                    "categorical": ["product_category", "warranty_period"],
                    "metrics": ["product_rating", "review_count"],
                },
            },
            "transactions": {
                "standard_columns": [
                    "transaction_id",
                    "customer_id",
                    "transaction_date",
                    "transaction_amount",
                    "transaction_type",
                    "payment_method",
                    "currency_code",
                    "merchant_id",
                    "authorization_code",
                    "transaction_status",
                    "fees",
                    "net_amount",
                    "card_type",
                    "card_last_four",
                    "processing_time",
                    "risk_score",
                    "terminal_id",
                    "batch_number",
                    "reference_number",
                    "chargeback_flag",
                ],
                "semantic_groups": {
                    "identifiers": [
                        "transaction_id",
                        "customer_id",
                        "merchant_id",
                        "terminal_id",
                    ],
                    "financial": [
                        "transaction_amount",
                        "fees",
                        "net_amount",
                        "risk_score",
                    ],
                    "payment": [
                        "payment_method",
                        "card_type",
                        "card_last_four",
                        "authorization_code",
                    ],
                    "processing": [
                        "processing_time",
                        "batch_number",
                        "reference_number",
                    ],
                    "status": ["transaction_status", "chargeback_flag"],
                    "temporal": ["transaction_date"],
                    "metadata": ["transaction_type", "currency_code"],
                },
            },
            "employees": {
                "standard_columns": [
                    "employee_id",
                    "employee_name",
                    "employee_email",
                    "department",
                    "job_title",
                    "hire_date",
                    "salary",
                    "manager_id",
                    "office_location",
                    "phone_extension",
                    "employment_status",
                    "termination_date",
                    "performance_rating",
                    "skill_level",
                    "education_level",
                    "years_experience",
                    "emergency_contact",
                    "benefits_eligible",
                    "vacation_days",
                    "sick_days",
                ],
                "semantic_groups": {
                    "identifiers": ["employee_id", "manager_id"],
                    "personal": [
                        "employee_name",
                        "employee_email",
                        "emergency_contact",
                    ],
                    "organizational": ["department", "job_title", "office_location"],
                    "financial": ["salary", "benefits_eligible"],
                    "temporal": ["hire_date", "termination_date"],
                    "performance": [
                        "performance_rating",
                        "skill_level",
                        "years_experience",
                    ],
                    "benefits": ["vacation_days", "sick_days"],
                    "status": ["employment_status"],
                    "contact": ["phone_extension"],
                },
            },
            # Additional domains for neural network richness
            "financial_accounts": {
                "standard_columns": [
                    "account_id",
                    "account_number",
                    "account_type",
                    "account_balance",
                    "account_status",
                    "opening_date",
                    "closing_date",
                    "interest_rate",
                    "minimum_balance",
                    "overdraft_limit",
                    "last_transaction_date",
                    "account_holder_name",
                    "branch_code",
                    "routing_number",
                    "iban",
                    "swift_code",
                    "account_currency",
                    "monthly_fee",
                    "annual_fee",
                ],
                "semantic_groups": {
                    "identifiers": [
                        "account_id",
                        "account_number",
                        "branch_code",
                        "routing_number",
                    ],
                    "financial": [
                        "account_balance",
                        "interest_rate",
                        "minimum_balance",
                        "overdraft_limit",
                    ],
                    "temporal": [
                        "opening_date",
                        "closing_date",
                        "last_transaction_date",
                    ],
                    "fees": ["monthly_fee", "annual_fee"],
                    "codes": ["swift_code", "iban"],
                    "metadata": ["account_type", "account_status", "account_currency"],
                },
            },
            "inventory": {
                "standard_columns": [
                    "item_id",
                    "item_name",
                    "item_code",
                    "warehouse_location",
                    "quantity_on_hand",
                    "quantity_reserved",
                    "quantity_available",
                    "reorder_point",
                    "reorder_quantity",
                    "lead_time_days",
                    "last_received_date",
                    "last_sold_date",
                    "unit_cost",
                    "selling_price",
                    "markup_percentage",
                    "item_category",
                    "item_subcategory",
                    "vendor_id",
                    "expiration_date",
                    "batch_number",
                ],
                "semantic_groups": {
                    "identifiers": [
                        "item_id",
                        "item_code",
                        "vendor_id",
                        "batch_number",
                    ],
                    "descriptive": ["item_name", "item_category", "item_subcategory"],
                    "quantities": [
                        "quantity_on_hand",
                        "quantity_reserved",
                        "quantity_available",
                    ],
                    "reorder": ["reorder_point", "reorder_quantity", "lead_time_days"],
                    "financial": ["unit_cost", "selling_price", "markup_percentage"],
                    "temporal": [
                        "last_received_date",
                        "last_sold_date",
                        "expiration_date",
                    ],
                    "location": ["warehouse_location"],
                },
            },
        }

    def _define_linguistic_patterns(self):
        """Define linguistic patterns for realistic variations"""
        return {
            "prefixes": [
                "src_",
                "tgt_",
                "dim_",
                "fact_",
                "stg_",
                "temp_",
                "old_",
                "new_",
                "tmp_",
                "bkp_",
                "arch_",
                "hist_",
                "curr_",
                "prev_",
                "next_",
                "raw_",
                "clean_",
                "valid_",
                "final_",
                "orig_",
                "master_",
            ],
            "suffixes": [
                "_id",
                "_key",
                "_code",
                "_num",
                "_number",
                "_ref",
                "_reference",
                "_val",
                "_value",
                "_amt",
                "_amount",
                "_qty",
                "_quantity",
                "_dt",
                "_date",
                "_time",
                "_ts",
                "_timestamp",
                "_flg",
                "_flag",
                "_ind",
                "_indicator",
                "_desc",
                "_description",
                "_nm",
                "_name",
                "_addr",
                "_address",
                "_info",
                "_data",
                "_details",
                "_spec",
            ],
            "separators": ["_", "-", ".", ""],
            "case_styles": ["lower", "upper", "title", "camel", "pascal"],
            "abbreviations": {
                "customer": ["cust", "client", "cli", "usr", "user", "acct", "account"],
                "product": ["prod", "item", "sku", "prd", "goods", "merchandise"],
                "order": ["ord", "purchase", "req", "request", "booking"],
                "transaction": ["trans", "txn", "trx", "payment", "pay"],
                "employee": ["emp", "staff", "personnel", "worker", "member"],
                "address": ["addr", "location", "loc", "place"],
                "telephone": ["tel", "phone", "mobile", "cell", "contact"],
                "email": ["mail", "e_mail", "electronic_mail"],
                "identifier": ["id", "key", "code", "ref", "number", "num"],
                "amount": ["amt", "sum", "total", "value", "cost", "price"],
                "quantity": ["qty", "count", "num", "volume", "units"],
                "description": ["desc", "info", "details", "notes", "comments"],
            },
            "international": {
                "customer_id": [
                    "cliente_id",
                    "kunden_id",
                    "client_id",
                    "utilisateur_id",
                ],
                "customer_name": [
                    "nome_cliente",
                    "kundenname",
                    "nom_client",
                    "customer_nombre",
                ],
                "product_name": [
                    "nome_produto",
                    "produktname",
                    "nom_produit",
                    "producto_nombre",
                ],
                "order_date": [
                    "data_pedido",
                    "bestelldatum",
                    "date_commande",
                    "fecha_pedido",
                ],
                "price": ["preco", "preis", "prix", "precio"],
                "address": ["endereco", "adresse", "direccion"],
            },
        }

    def _define_augmentation_strategies(self):
        """Define data augmentation strategies"""
        return {
            "typos": {
                "common_typos": ["teh", "adn", "hte", "nad"],
                "character_swaps": True,
                "missing_characters": True,
                "extra_characters": True,
            },
            "noise": {
                "random_numbers": True,
                "random_characters": True,
                "version_numbers": ["_v1", "_v2", "_2023", "_old", "_new"],
            },
            "formatting": {
                "extra_spaces": True,
                "mixed_case": True,
                "special_characters": ["@", "#", "$", "%", "&", "*"],
            },
        }

    def generate_base_variations(self, standard_col, domain_info):
        """Generate base variations for a standard column"""
        variations = set()

        # Add the original
        variations.add(standard_col)

        # Semantic-based variations using abbreviations
        for word, abbrevs in self.linguistic_patterns["abbreviations"].items():
            if word in standard_col.lower():
                for abbrev in abbrevs:
                    # Replace full word with abbreviation
                    variation = standard_col.lower().replace(word, abbrev)
                    variations.add(variation)

                    # Also try without underscores
                    variations.add(variation.replace("_", ""))

        # Add prefix/suffix combinations
        base_col = standard_col
        for prefix in self.linguistic_patterns["prefixes"][
            :10
        ]:  # Limit for base generation
            variations.add(f"{prefix}{base_col}")

        for suffix in self.linguistic_patterns["suffixes"][
            :10
        ]:  # Limit for base generation
            variations.add(f"{base_col}{suffix}")

        # Case variations
        variations.add(standard_col.upper())
        variations.add(standard_col.lower())
        variations.add(standard_col.capitalize())

        # Separator variations
        if "_" in standard_col:
            variations.add(standard_col.replace("_", ""))
            variations.add(standard_col.replace("_", "-"))
            variations.add(standard_col.replace("_", "."))

        # International variations if available
        if standard_col in self.linguistic_patterns["international"]:
            variations.update(self.linguistic_patterns["international"][standard_col])

        return list(variations)

    def apply_augmentation(self, base_variations, augmentation_factor=5):
        """Apply data augmentation to create more variations"""
        augmented = []

        for base_var in base_variations:
            augmented.append(base_var)  # Keep original

            # Generate multiple augmented versions
            for _ in range(augmentation_factor):
                augmented_var = base_var

                # Apply typos (20% chance)
                if random.random() < 0.2:
                    augmented_var = self._introduce_typos(augmented_var)

                # Apply noise (15% chance)
                if random.random() < 0.15:
                    augmented_var = self._add_noise(augmented_var)

                # Apply formatting changes (30% chance)
                if random.random() < 0.3:
                    augmented_var = self._apply_formatting_changes(augmented_var)

                # Apply additional prefixes/suffixes (25% chance)
                if random.random() < 0.25:
                    if random.random() < 0.5:
                        prefix = random.choice(self.linguistic_patterns["prefixes"])
                        augmented_var = f"{prefix}{augmented_var}"
                    else:
                        suffix = random.choice(self.linguistic_patterns["suffixes"])
                        augmented_var = f"{augmented_var}{suffix}"

                augmented.append(augmented_var)

        # Remove duplicates and return
        return list(set(augmented))

    def _introduce_typos(self, text):
        """Introduce realistic typos"""
        if len(text) < 3:
            return text

        typo_type = random.choice(["swap", "missing", "extra", "substitute"])

        if typo_type == "swap" and len(text) > 3:
            # Swap two adjacent characters
            pos = random.randint(0, len(text) - 2)
            text_list = list(text)
            text_list[pos], text_list[pos + 1] = text_list[pos + 1], text_list[pos]
            return "".join(text_list)

        elif typo_type == "missing" and len(text) > 3:
            # Remove a character
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos + 1 :]

        elif typo_type == "extra":
            # Add an extra character
            pos = random.randint(0, len(text))
            char = random.choice(string.ascii_lowercase)
            return text[:pos] + char + text[pos:]

        elif typo_type == "substitute":
            # Substitute a character
            pos = random.randint(0, len(text) - 1)
            char = random.choice(string.ascii_lowercase)
            return text[:pos] + char + text[pos + 1 :]

        return text

    def _add_noise(self, text):
        """Add realistic noise patterns"""
        noise_type = random.choice(["numbers", "version", "system"])

        if noise_type == "numbers":
            # Add random numbers
            number = random.randint(1, 999)
            position = random.choice(["prefix", "suffix"])
            if position == "prefix":
                return f"{number}_{text}"
            else:
                return f"{text}_{number}"

        elif noise_type == "version":
            # Add version information
            version = random.choice(
                self.augmentation_strategies["noise"]["version_numbers"]
            )
            return f"{text}{version}"

        elif noise_type == "system":
            # Add system prefixes
            system = random.choice(["sys", "app", "db", "tbl"])
            return f"{system}_{text}"

        return text

    def _apply_formatting_changes(self, text):
        """Apply formatting variations"""
        # Case changes
        case_type = random.choice(["upper", "lower", "title", "mixed"])

        if case_type == "upper":
            text = text.upper()
        elif case_type == "lower":
            text = text.lower()
        elif case_type == "title":
            text = text.title()
        elif case_type == "mixed":
            # Randomly capitalize characters
            text = "".join(
                c.upper() if random.random() < 0.3 else c.lower() for c in text
            )

        # Separator changes
        if random.random() < 0.4:
            old_sep = "_"
            new_sep = random.choice(["-", ".", "", " "])
            text = text.replace(old_sep, new_sep)

        return text

    def generate_cross_domain_negatives(
        self, raw_col, correct_domain, correct_standard, num_negatives=8
    ):
        """Generate challenging cross-domain negative examples"""
        negatives = []

        # Get all other domains and their columns
        other_domains = {k: v for k, v in self.domains.items() if k != correct_domain}

        # Strategy 1: Same semantic group, different domain
        correct_semantic_group = None
        for group, columns in self.domains[correct_domain]["semantic_groups"].items():
            if correct_standard in columns:
                correct_semantic_group = group
                break

        if correct_semantic_group:
            for domain, info in other_domains.items():
                if correct_semantic_group in info["semantic_groups"]:
                    candidates = info["semantic_groups"][correct_semantic_group]
                    for candidate in candidates[:2]:  # Limit per domain
                        confidence = random.uniform(
                            0.3, 0.7
                        )  # Medium confidence for hard negatives
                        negatives.append(
                            {
                                "table_name": domain,
                                "standard_column_name": candidate,
                                "raw_column_name": raw_col,
                                "is_match": False,
                                "confidence_score": confidence,
                                "negative_type": "semantic_similar",
                            }
                        )

        # Strategy 2: Lexically similar columns
        for domain, info in other_domains.items():
            for std_col in info["standard_columns"]:
                # Check lexical similarity
                if self._lexical_similarity(correct_standard, std_col) > 0.5:
                    confidence = random.uniform(0.2, 0.6)
                    negatives.append(
                        {
                            "table_name": domain,
                            "standard_column_name": std_col,
                            "raw_column_name": raw_col,
                            "is_match": False,
                            "confidence_score": confidence,
                            "negative_type": "lexical_similar",
                        }
                    )

        # Strategy 3: Random negatives (easy)
        all_other_columns = []
        for domain, info in other_domains.items():
            all_other_columns.extend(
                [(domain, col) for col in info["standard_columns"]]
            )

        random.shuffle(all_other_columns)
        for domain, std_col in all_other_columns[: num_negatives // 3]:
            confidence = random.uniform(
                0.1, 0.4
            )  # Low confidence for obvious wrong matches
            negatives.append(
                {
                    "table_name": domain,
                    "standard_column_name": std_col,
                    "raw_column_name": raw_col,
                    "is_match": False,
                    "confidence_score": confidence,
                    "negative_type": "random",
                }
            )

        # Limit and shuffle
        random.shuffle(negatives)
        return negatives[:num_negatives]

    def _lexical_similarity(self, str1, str2):
        """Calculate simple lexical similarity"""
        set1 = set(str1.lower().split("_"))
        set2 = set(str2.lower().split("_"))

        if not set1 or not set2:
            return 0

        return len(set1.intersection(set2)) / len(set1.union(set2))

    def generate_production_dataset(self):
        """Generate large-scale production dataset for neural networks"""
        print(f"Generating production dataset with target size: {self.target_size:,}")

        all_examples = []
        examples_per_domain = defaultdict(int)

        # Calculate target examples per standard column
        total_standard_columns = sum(
            len(info["standard_columns"]) for info in self.domains.values()
        )
        examples_per_column = max(
            1000, self.target_size // (total_standard_columns * 6)
        )  # 6 = 1 pos + 5 neg avg

        print(f"Target examples per standard column: {examples_per_column:,}")

        for domain_name, domain_info in self.domains.items():
            print(f"\nProcessing domain: {domain_name}")

            for standard_col in domain_info["standard_columns"]:
                # Generate base variations
                base_variations = self.generate_base_variations(
                    standard_col, domain_info
                )

                # Apply augmentation
                all_variations = self.apply_augmentation(
                    base_variations, augmentation_factor=10
                )

                # Limit variations per column to control dataset size
                if len(all_variations) > examples_per_column:
                    all_variations = random.sample(all_variations, examples_per_column)

                # Generate positive examples
                for raw_col in all_variations:
                    confidence = random.uniform(0.85, 0.99)
                    all_examples.append(
                        {
                            "table_name": domain_name,
                            "standard_column_name": standard_col,
                            "raw_column_name": raw_col,
                            "is_match": True,
                            "confidence_score": confidence,
                            "example_type": "positive",
                        }
                    )
                    examples_per_domain[domain_name] += 1

                # Generate negative examples for each raw column variation
                for raw_col in random.sample(
                    all_variations, min(len(all_variations), examples_per_column // 3)
                ):
                    negatives = self.generate_cross_domain_negatives(
                        raw_col, domain_name, standard_col, num_negatives=5
                    )
                    all_examples.extend(negatives)
                    examples_per_domain[domain_name] += len(negatives)

        # Additional hard negatives within same domain
        print("\nGenerating intra-domain hard negatives...")
        intra_domain_negatives = self._generate_intra_domain_negatives(all_examples)
        all_examples.extend(intra_domain_negatives)

        # Shuffle the dataset
        random.shuffle(all_examples)

        # Create DataFrame
        df = pd.DataFrame(all_examples)

        # Print statistics
        print(f"\n{'='*50}")
        print("DATASET GENERATION COMPLETE")
        print(f"{'='*50}")
        print(f"Total examples generated: {len(df):,}")
        print(f"Positive examples: {len(df[df['is_match'] == True]):,}")
        print(f"Negative examples: {len(df[df['is_match'] == False]):,}")
        print(f"Positive ratio: {len(df[df['is_match'] == True]) / len(df):.1%}")

        print(f"\nExamples per domain:")
        for domain, count in sorted(examples_per_domain.items()):
            print(f"  {domain}: {count:,}")

        print(f"\nUnique standard columns: {df['standard_column_name'].nunique()}")
        print(f"Unique raw columns: {df['raw_column_name'].nunique()}")
        print(
            f"Average examples per standard column: {len(df) / df['standard_column_name'].nunique():.1f}"
        )

        if "negative_type" in df.columns:
            print(f"\nNegative example types:")
            neg_types = df[df["is_match"] == False]["negative_type"].value_counts()
            for neg_type, count in neg_types.items():
                print(f"  {neg_type}: {count:,}")

        return df

    def _generate_intra_domain_negatives(self, existing_examples):
        """Generate hard negatives within the same domain"""
        intra_negatives = []

        # Group positive examples by domain
        positive_examples = [ex for ex in existing_examples if ex["is_match"]]
        domain_examples = defaultdict(list)

        for ex in positive_examples:
            domain_examples[ex["table_name"]].append(ex)

        # For each domain, create wrong mappings within domain
        for domain, examples in domain_examples.items():
            domain_columns = list(set(ex["standard_column_name"] for ex in examples))

            # Sample some examples for intra-domain negatives
            sampled_examples = random.sample(
                examples, min(len(examples), 500)
            )  # Limit for performance

            for ex in sampled_examples:
                raw_col = ex["raw_column_name"]
                correct_standard = ex["standard_column_name"]

                # Pick wrong columns from same domain
                wrong_columns = [
                    col for col in domain_columns if col != correct_standard
                ]
                selected_wrong = random.sample(
                    wrong_columns, min(3, len(wrong_columns))
                )

                for wrong_col in selected_wrong:
                    # Higher confidence for lexically similar wrong matches (harder negatives)
                    if self._lexical_similarity(correct_standard, wrong_col) > 0.3:
                        confidence = random.uniform(0.4, 0.8)
                        neg_type = "intra_domain_hard"
                    else:
                        confidence = random.uniform(0.1, 0.5)
                        neg_type = "intra_domain_easy"

                    intra_negatives.append(
                        {
                            "table_name": domain,
                            "standard_column_name": wrong_col,
                            "raw_column_name": raw_col,
                            "is_match": False,
                            "confidence_score": confidence,
                            "negative_type": neg_type,
                        }
                    )

        print(f"Generated {len(intra_negatives):,} intra-domain negative examples")
        return intra_negatives

    def save_dataset(self, df, filename="neural_column_mapping_dataset"):
        """Save dataset in multiple formats for neural network training"""

        # Save as Excel (matching your format)
        excel_file = f"{filename}.xlsx"
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            # Save by domain (matching your original format)
            for domain in df["table_name"].unique():
                domain_df = df[df["table_name"] == domain].copy()
                # Keep only required columns for training
                training_df = domain_df[
                    [
                        "standard_column_name",
                        "raw_column_name",
                        "is_match",
                        "confidence_score",
                    ]
                ]
                training_df.to_excel(writer, sheet_name=domain, index=False)

            # Master sheet with all data
            df.to_excel(writer, sheet_name="all_data", index=False)

        print(f"Excel dataset saved: {excel_file}")

        # Save as CSV for easier neural network loading
        csv_file = f"{filename}.csv"
        df.to_csv(csv_file, index=False)
        print(f"CSV dataset saved: {csv_file}")

        # Save as JSON for metadata preservation
        json_file = f"{filename}_metadata.json"
        metadata = {
            "total_examples": len(df),
            "positive_examples": len(df[df["is_match"] == True]),
            "negative_examples": len(df[df["is_match"] == False]),
            "domains": list(df["table_name"].unique()),
            "standard_columns": list(df["standard_column_name"].unique()),
            "unique_raw_columns": df["raw_column_name"].nunique(),
            "generation_config": {
                "target_size": self.target_size,
                "domains_used": len(self.domains),
                "augmentation_applied": True,
            },
        }

        with open(json_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved: {json_file}")

        # Save train/validation/test splits for neural network training
        self._save_neural_splits(df, filename)

        return excel_file, csv_file, json_file

    def _save_neural_splits(self, df, filename):
        """Create and save train/validation/test splits optimized for neural networks"""

        # Stratified split by standard column to ensure all columns in each split
        # First split: 80% train, 20% temp
        train_df, temp_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["standard_column_name"]
        )

        # Second split: 10% validation, 10% test from temp
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df["standard_column_name"],
        )

        # Save splits
        train_df.to_csv(f"{filename}_train.csv", index=False)
        val_df.to_csv(f"{filename}_val.csv", index=False)
        test_df.to_csv(f"{filename}_test.csv", index=False)

        print(f"Neural network splits saved:")
        print(f"  Training: {len(train_df):,} examples ({len(train_df)/len(df):.1%})")
        print(f"  Validation: {len(val_df):,} examples ({len(val_df)/len(df):.1%})")
        print(f"  Test: {len(test_df):,} examples ({len(test_df)/len(df):.1%})")

    def create_contrastive_pairs(self, df, output_file=None):
        """Create contrastive learning pairs for Siamese network training"""
        print("Creating contrastive pairs for Siamese network training...")

        positive_pairs = []
        negative_pairs = []

        # Get positive examples
        positive_examples = df[df["is_match"] == True].copy()

        # Create positive pairs (same standard column, different raw columns)
        print("Creating positive pairs...")
        for standard_col in positive_examples["standard_column_name"].unique():
            raw_columns = positive_examples[
                positive_examples["standard_column_name"] == standard_col
            ]["raw_column_name"].tolist()

            # Create pairs from raw columns mapping to same standard
            if len(raw_columns) > 1:
                for i, raw1 in enumerate(raw_columns[:20]):  # Limit combinations
                    for raw2 in raw_columns[i + 1 : 20]:
                        positive_pairs.append(
                            {
                                "raw_column_1": raw1,
                                "raw_column_2": raw2,
                                "standard_column": standard_col,
                                "label": 1,  # Similar
                                "confidence": random.uniform(0.85, 0.99),
                            }
                        )

        # Create negative pairs (different standard columns)
        print("Creating negative pairs...")
        standard_columns = list(positive_examples["standard_column_name"].unique())

        # Sample for negative pairs to control size
        sample_size = min(
            len(positive_pairs) * 2, 50000
        )  # 2:1 negative to positive ratio

        for _ in range(sample_size):
            # Pick two different standard columns
            std1, std2 = random.sample(standard_columns, 2)

            # Get raw columns for each
            raw1_options = positive_examples[
                positive_examples["standard_column_name"] == std1
            ]["raw_column_name"].tolist()

            raw2_options = positive_examples[
                positive_examples["standard_column_name"] == std2
            ]["raw_column_name"].tolist()

            if raw1_options and raw2_options:
                raw1 = random.choice(raw1_options)
                raw2 = random.choice(raw2_options)

                # Calculate difficulty-based confidence
                similarity = self._lexical_similarity(std1, std2)
                if similarity > 0.3:
                    confidence = random.uniform(0.3, 0.7)  # Hard negative
                else:
                    confidence = random.uniform(0.1, 0.4)  # Easy negative

                negative_pairs.append(
                    {
                        "raw_column_1": raw1,
                        "raw_column_2": raw2,
                        "standard_column_1": std1,
                        "standard_column_2": std2,
                        "label": 0,  # Different
                        "confidence": confidence,
                    }
                )

        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)

        contrastive_df = pd.DataFrame(all_pairs)

        if output_file:
            contrastive_df.to_csv(output_file, index=False)
            print(f"Contrastive pairs saved to: {output_file}")

        print(f"Created {len(positive_pairs):,} positive pairs")
        print(f"Created {len(negative_pairs):,} negative pairs")
        print(f"Total contrastive pairs: {len(all_pairs):,}")

        return contrastive_df

    def generate_few_shot_examples(self, df, shots_per_column=5):
        """Generate few-shot learning examples for meta-learning approaches"""
        print(f"Creating few-shot examples with {shots_per_column} shots per column...")

        few_shot_data = defaultdict(list)

        positive_examples = df[df["is_match"] == True].copy()

        for standard_col in positive_examples["standard_column_name"].unique():
            col_examples = positive_examples[
                positive_examples["standard_column_name"] == standard_col
            ]

            # Sample few-shot examples
            if len(col_examples) >= shots_per_column:
                support_examples = col_examples.sample(
                    n=shots_per_column, random_state=42
                )
                query_examples = col_examples.drop(support_examples.index)

                few_shot_data["support"].extend(
                    [
                        {
                            "standard_column": standard_col,
                            "raw_column": row["raw_column_name"],
                            "table_name": row["table_name"],
                            "shot_id": i,
                        }
                        for i, (_, row) in enumerate(support_examples.iterrows())
                    ]
                )

                few_shot_data["query"].extend(
                    [
                        {
                            "standard_column": standard_col,
                            "raw_column": row["raw_column_name"],
                            "table_name": row["table_name"],
                            "is_match": True,
                        }
                        for _, row in query_examples.iterrows()
                    ]
                )

        return few_shot_data


def main():
    """Main function to generate production neural network dataset"""
    print("=" * 60)
    print("NEURAL NETWORK COLUMN MAPPING DATASET GENERATOR")
    print("=" * 60)

    # Configuration
    target_sizes = {
        "small": 50000,  # For experimentation
        "medium": 200000,  # Recommended for production
        "large": 500000,  # For maximum performance
        "xl": 1000000,  # For large-scale deployment
    }

    print("\nAvailable dataset sizes:")
    for size_name, size_val in target_sizes.items():
        print(f"  {size_name}: {size_val:,} examples")

    # Get user choice or default to medium
    size_choice = (
        input("\nEnter size choice (small/medium/large/xl) [default: medium]: ")
        .strip()
        .lower()
    )
    if size_choice not in target_sizes:
        size_choice = "medium"

    target_size = target_sizes[size_choice]
    print(f"Generating {size_choice} dataset with {target_size:,} target examples...")

    os.makedirs("./data", exist_ok=True)

    # Initialize generator
    generator = NeuralDatasetGenerator(target_size=target_size)

    # Generate main dataset
    print("\n" + "=" * 50)
    print("PHASE 1: GENERATING MAIN DATASET")
    print("=" * 50)

    df = generator.generate_production_dataset()

    # Save dataset
    print("\n" + "=" * 50)
    print("PHASE 2: SAVING DATASETS")
    print("=" * 50)

    filename = f"./data/neural_column_mapping_{size_choice}"
    excel_file, csv_file, json_file = generator.save_dataset(df, filename)

    # Generate contrastive pairs
    print("\n" + "=" * 50)
    print("PHASE 3: GENERATING CONTRASTIVE PAIRS")
    print("=" * 50)

    contrastive_file = f"{filename}_contrastive.csv"
    contrastive_df = generator.create_contrastive_pairs(df, contrastive_file)

    # Generate few-shot data
    print("\n" + "=" * 50)
    print("PHASE 4: GENERATING FEW-SHOT EXAMPLES")
    print("=" * 50)

    few_shot_data = generator.generate_few_shot_examples(df)
    few_shot_file = f"{filename}_few_shot.json"
    with open(few_shot_file, "w") as f:
        json.dump(few_shot_data, f, indent=2)
    print(f"Few-shot data saved to: {few_shot_file}")

    # Final summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print(f"\nFiles generated:")
    print(f"  📊 Main dataset (Excel): {excel_file}")
    print(f"  📊 Main dataset (CSV): {csv_file}")
    print(f"  📋 Metadata: {json_file}")
    print(f"  🔄 Contrastive pairs: {contrastive_file}")
    print(f"  🎯 Few-shot examples: {few_shot_file}")
    print(
        f"  🧠 Neural splits: {filename}_train.csv, {filename}_val.csv, {filename}_test.csv"
    )

    print(f"\nDataset Statistics:")
    print(f"  Total examples: {len(df):,}")
    print(f"  Positive examples: {len(df[df['is_match'] == True]):,}")
    print(f"  Negative examples: {len(df[df['is_match'] == False]):,}")
    print(f"  Unique standard columns: {df['standard_column_name'].nunique()}")
    print(f"  Unique raw columns: {df['raw_column_name'].nunique()}")
    print(f"  Domains covered: {df['table_name'].nunique()}")

    print(f"\nReady for neural network training! 🚀")

    return df, generator


# Example usage functions
def quick_generate(size="medium"):
    """Quick generate function for easy use"""
    target_sizes = {"small": 50000, "medium": 200000, "large": 500000, "xl": 1000000}

    generator = NeuralDatasetGenerator(target_size=target_sizes[size])
    df = generator.generate_production_dataset()

    filename = f"./data/neural_column_mapping_{size}"
    generator.save_dataset(df, filename)

    return df, generator


def generate_for_specific_domains(domains_to_include, target_size=100000):
    """Generate dataset for specific domains only"""
    generator = NeuralDatasetGenerator(target_size=target_size)

    # Filter domains
    filtered_domains = {
        k: v for k, v in generator.domains.items() if k in domains_to_include
    }
    generator.domains = filtered_domains

    df = generator.generate_production_dataset()

    filename = f"./data/neural_column_mapping_{'_'.join(domains_to_include)}"
    generator.save_dataset(df, filename)

    return df, generator


if __name__ == "__main__":
    # Run the main dataset generation
    dataset, generator = main()
