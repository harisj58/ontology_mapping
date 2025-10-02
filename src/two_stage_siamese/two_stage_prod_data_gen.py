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
            "fact": {
                "standard_columns": [
                    "GEOGRAPHY_KEY_TOTAL",
                    "TIME_KEY_TOTAL",
                    "UPC_13_DIGIT_TOTAL",
                    "DOLLAR_SALES_TOTAL",
                    "DOLLAR_SALES_ANY_MERCH",
                    "DOLLAR_SALES_PRICE_REDUCTIONS_ONLY",
                    "DOLLAR_SALES_FEATURE_ONLY",
                    "DOLLAR_SALES_DISPLAY_ONLY",
                    "DOLLAR_SALES_FEATURE_AND_DISPLAY",
                    "UNIT_SALES_TOTAL",
                    "UNIT_SALES_ANY_MERCH",
                    "UNIT_SALES_PRICE_REDUCTIONS_ONLY",
                    "UNIT_SALES_FEATURE_ONLY",
                    "UNIT_SALES_DISPLAY_ONLY",
                    "UNIT_SALES_FEATURE_AND_DISPLAY",
                    "VOLUME_SALES_TOTAL",
                    "VOLUME_SALES_ANY_MERCH",
                    "VOLUME_SALES_PRICE_REDUCTIONS_ONLY",
                    "VOLUME_SALES_FEATURE_ONLY",
                    "VOLUME_SALES_DISPLAY_ONLY",
                    "VOLUME_SALES_FEATURE_AND_DISPLAY",
                    "BASE_DOLLAR_SALES_TOTAL",
                    "BASE_UNIT_SALES_TOTAL",
                    "BASE_VOLUME_SALES_TOTAL",
                    "ACV_WEIGHTED_DISTRIBUTION_TOTAL",
                    "AVERAGE_WEEKLY_ACV_DISTRIBUTION_TOTAL",
                    "AVERAGE_WEEKLY_ACV_DISTRIBUTION_ANY_MERCH",
                    "AVERAGE_WEEKLY_ACV_DISTRIBUTION_PRICE_REDUCTIONS_ONLY",
                    "AVERAGE_WEEKLY_ACV_DISTRIBUTION_FEATURE_ONLY",
                    "AVERAGE_WEEKLY_ACV_DISTRIBUTION_DISPLAY_ONLY",
                    "AVERAGE_WEEKLY_ACV_DISTRIBUTION_FEATURE_AND_DISPLAY",
                    "INCREMENTAL_VOLUME_TOTAL",
                    "INCREMENTAL_VOLUME_PRICE_REDUCTIONS_ONLY",
                    "INCREMENTAL_VOLUME_FEATURE_ONLY",
                    "INCREMENTAL_VOLUME_DISPLAY_ONLY",
                    "INCREMENTAL_VOLUME_FEATURE_AND_DISPLAY",
                    "WEIGHTED_AVERAGE_BASE_PRICE_PER_VOLUME_TOTAL",
                    "CATEGORY_WEIGHTED_DISTRIBUTION_TOTAL",
                    "TOTAL_POINTS_OF_CATEGORY_WEIGHTED_DISTRIBUTION_TOTAL",
                    "DISCOUNT_TOTAL",
                ],
                "semantic_groups": {
                    "identifiers": [
                        "GEOGRAPHY_KEY_TOTAL",
                        "TIME_KEY_TOTAL",
                        "UPC_13_DIGIT_TOTAL",
                    ],
                    "sales_dollar": [
                        "DOLLAR_SALES_TOTAL",
                        "DOLLAR_SALES_ANY_MERCH",
                        "DOLLAR_SALES_PRICE_REDUCTIONS_ONLY",
                        "DOLLAR_SALES_FEATURE_ONLY",
                        "DOLLAR_SALES_DISPLAY_ONLY",
                        "DOLLAR_SALES_FEATURE_AND_DISPLAY",
                        "BASE_DOLLAR_SALES_TOTAL",
                    ],
                    "sales_unit": [
                        "UNIT_SALES_TOTAL",
                        "UNIT_SALES_ANY_MERCH",
                        "UNIT_SALES_PRICE_REDUCTIONS_ONLY",
                        "UNIT_SALES_FEATURE_ONLY",
                        "UNIT_SALES_DISPLAY_ONLY",
                        "UNIT_SALES_FEATURE_AND_DISPLAY",
                        "BASE_UNIT_SALES_TOTAL",
                    ],
                    "sales_volume": [
                        "VOLUME_SALES_TOTAL",
                        "VOLUME_SALES_ANY_MERCH",
                        "VOLUME_SALES_PRICE_REDUCTIONS_ONLY",
                        "VOLUME_SALES_FEATURE_ONLY",
                        "VOLUME_SALES_DISPLAY_ONLY",
                        "VOLUME_SALES_FEATURE_AND_DISPLAY",
                        "BASE_VOLUME_SALES_TOTAL",
                    ],
                    "distribution": [
                        "ACV_WEIGHTED_DISTRIBUTION_TOTAL",
                        "AVERAGE_WEEKLY_ACV_DISTRIBUTION_TOTAL",
                        "AVERAGE_WEEKLY_ACV_DISTRIBUTION_ANY_MERCH",
                        "AVERAGE_WEEKLY_ACV_DISTRIBUTION_PRICE_REDUCTIONS_ONLY",
                        "AVERAGE_WEEKLY_ACV_DISTRIBUTION_FEATURE_ONLY",
                        "AVERAGE_WEEKLY_ACV_DISTRIBUTION_DISPLAY_ONLY",
                        "AVERAGE_WEEKLY_ACV_DISTRIBUTION_FEATURE_AND_DISPLAY",
                        "CATEGORY_WEIGHTED_DISTRIBUTION_TOTAL",
                        "TOTAL_POINTS_OF_CATEGORY_WEIGHTED_DISTRIBUTION_TOTAL",
                    ],
                    "incremental_metrics": [
                        "INCREMENTAL_VOLUME_TOTAL",
                        "INCREMENTAL_VOLUME_PRICE_REDUCTIONS_ONLY",
                        "INCREMENTAL_VOLUME_FEATURE_ONLY",
                        "INCREMENTAL_VOLUME_DISPLAY_ONLY",
                        "INCREMENTAL_VOLUME_FEATURE_AND_DISPLAY",
                    ],
                    "pricing": [
                        "WEIGHTED_AVERAGE_BASE_PRICE_PER_VOLUME_TOTAL",
                        "DISCOUNT_TOTAL",
                    ],
                },
            },
            "geography_dim": {
                "standard_columns": ["GEOGRAPHY_KEY", "GEOGRAPHY_DESCRIPTION"],
                "semantic_groups": {
                    "identifiers": ["GEOGRAPHY_KEY"],
                    "descriptions": ["GEOGRAPHY_DESCRIPTION"],
                },
            },
            "product_dim": {
                "standard_columns": [
                    "PRODUCT_DESCRIPTION",
                    "UPC_13_DIGIT",
                    "ALTERNATIVE_ADULT_BEVERAGE_VALUE",
                    "BEER_CATEGORY_VALUE",
                    "BEER_VENDOR_VALUE",
                    "BEER_SEGMENT_VALUE",
                    "BEER_PRICE_SEGMENT_VALUE",
                    "BEER_BRAND_FAMILY_VALUE",
                    "BEER_BRAND_VALUE",
                    "BEER_DOM_VS_IMP_VALUE",
                    "BEER_PACKAGE_VALUE",
                    "BEER_SIZE_VALUE",
                    "BEER_BRAND_TYPE_VALUE",
                    "BEER_SIZE_GROUP_VALUE",
                    "MEZZO_STYLE_VALUE",
                    "MACRO_STYLE_VALUE",
                    "MICRO_STYLE_VALUE",
                    "BEER_COUNT_GROUP_VALUE",
                ],
                "semantic_groups": {
                    "identifiers": ["UPC_13_DIGIT"],
                    "descriptions": ["PRODUCT_DESCRIPTION"],
                    "category_attributes": [
                        "ALTERNATIVE_ADULT_BEVERAGE_VALUE",
                        "BEER_CATEGORY_VALUE",
                        "BEER_SEGMENT_VALUE",
                        "BEER_PRICE_SEGMENT_VALUE",
                    ],
                    "brand_attributes": [
                        "BEER_VENDOR_VALUE",
                        "BEER_BRAND_FAMILY_VALUE",
                        "BEER_BRAND_VALUE",
                        "BEER_BRAND_TYPE_VALUE",
                        "BEER_DOM_VS_IMP_VALUE",
                    ],
                    "packaging_attributes": [
                        "BEER_PACKAGE_VALUE",
                        "BEER_SIZE_VALUE",
                        "BEER_SIZE_GROUP_VALUE",
                        "BEER_COUNT_GROUP_VALUE",
                    ],
                    "style_attributes": [
                        "MEZZO_STYLE_VALUE",
                        "MACRO_STYLE_VALUE",
                        "MICRO_STYLE_VALUE",
                    ],
                },
            },
            "time_dim": {
                "standard_columns": ["TIME_KEY", "TIME_DESCRIPTION"],
                "semantic_groups": {
                    "identifiers": ["TIME_KEY"],
                    "descriptions": ["TIME_DESCRIPTION"],
                },
            },
        }

    def _define_file_patterns(self):
        """Define file naming patterns for each table"""
        return {
            "fact": {
                "explicit": [
                    "fact.{ext}",
                    "fact_table.{ext}",
                    "sales_fact.{ext}",
                    "fact_export.{ext}",
                    "fact_{date}.{ext}",
                    "fact_metrics.{ext}",
                    "raw_fact.{ext}",
                    "staging_fact.{ext}",
                    "fact_master.{ext}",
                    "fact_{version}.{ext}",
                    "fact_dump.{ext}",
                    "fact_snapshot_{date}.{ext}",
                ],
                "implicit": [
                    "sales_data.{ext}",
                    "transaction_data.{ext}",
                    "market_facts.{ext}",
                    "sales_export.{ext}",
                    "performance_metrics.{ext}",
                    "revenue_data.{ext}",
                    "kpi_export.{ext}",
                    "financials.{ext}",
                    "analysis_facts.{ext}",
                    "business_metrics.{ext}",
                ],
                "ambiguous": [
                    "data.{ext}",
                    "export_{date}.{ext}",
                    "report.{ext}",
                    "dataset.{ext}",
                    "output_file.{ext}",
                    "records.{ext}",
                    "dump.{ext}",
                    "details.{ext}",
                    "datafile.{ext}",
                    "snapshot.{ext}",
                ],
                "abbreviations": [
                    "fct.{ext}",
                    "facts.{ext}",
                    "sfct.{ext}",
                    "fact_tbl.{ext}",
                    "facttbl.{ext}",
                    "f_table.{ext}",
                    "ft.{ext}",
                    "fct_data.{ext}",
                    "fct_exp.{ext}",
                    "fct_mstr.{ext}",
                    "fct_dim.{ext}",
                    "fct_snp.{ext}",
                ],
            },
            "geography_dim": {
                "explicit": [
                    "geography_dim.{ext}",
                    "geography_dimension.{ext}",
                    "geo_dim.{ext}",
                    "geography_export.{ext}",
                    "geography_{date}.{ext}",
                    "geography_master.{ext}",
                    "geo_dimension.{ext}",
                    "geography_snapshot.{ext}",
                    "staging_geography.{ext}",
                    "geo_lookup.{ext}",
                ],
                "implicit": [
                    "locations.{ext}",
                    "region_data.{ext}",
                    "territories.{ext}",
                    "area_mapping.{ext}",
                    "location_master.{ext}",
                    "regional_dim.{ext}",
                    "geo_data.{ext}",
                    "zones.{ext}",
                    "country_dim.{ext}",
                    "state_region.{ext}",
                ],
                "ambiguous": [
                    "data.{ext}",
                    "mapping.{ext}",
                    "lookup.{ext}",
                    "dim_export.{ext}",
                    "master_file.{ext}",
                    "ref_data.{ext}",
                    "codes.{ext}",
                    "reference.{ext}",
                    "list.{ext}",
                    "dim_dump.{ext}",
                ],
                "abbreviations": [
                    "geo.{ext}",
                    "geog.{ext}",
                    "gdim.{ext}",
                    "gd.{ext}",
                    "geo_dim.{ext}",
                    "geo_ref.{ext}",
                    "geo_tbl.{ext}",
                    "geo_mstr.{ext}",
                    "geo_lkp.{ext}",
                    "geo_snp.{ext}",
                    "g_data.{ext}",
                    "geo_exp.{ext}",
                ],
            },
            "product_dim": {
                "explicit": [
                    "product_dim.{ext}",
                    "product_dimension.{ext}",
                    "prod_dim.{ext}",
                    "product_export.{ext}",
                    "product_{date}.{ext}",
                    "product_master.{ext}",
                    "item_dim.{ext}",
                    "product_snapshot.{ext}",
                    "product_lookup.{ext}",
                    "staging_product.{ext}",
                    "product_catalog.{ext}",
                    "product_list.{ext}",
                ],
                "implicit": [
                    "items.{ext}",
                    "sku_data.{ext}",
                    "merchandise.{ext}",
                    "inventory.{ext}",
                    "catalog.{ext}",
                    "assortment.{ext}",
                    "brand_dim.{ext}",
                    "product_hierarchy.{ext}",
                    "sku_master.{ext}",
                    "item_data.{ext}",
                ],
                "ambiguous": [
                    "data.{ext}",
                    "export_{date}.{ext}",
                    "dump.{ext}",
                    "records.{ext}",
                    "dataset.{ext}",
                    "master_file.{ext}",
                    "details.{ext}",
                    "upload.{ext}",
                    "report.{ext}",
                    "ref_data.{ext}",
                ],
                "abbreviations": [
                    "prd.{ext}",
                    "prod.{ext}",
                    "pd.{ext}",
                    "pdim.{ext}",
                    "prd_dim.{ext}",
                    "prd_tbl.{ext}",
                    "prd_ref.{ext}",
                    "prd_mstr.{ext}",
                    "prd_exp.{ext}",
                    "prd_snp.{ext}",
                    "prd_lkp.{ext}",
                    "prd_cat.{ext}",
                ],
            },
            "time_dim": {
                "explicit": [
                    "time_dim.{ext}",
                    "time_dimension.{ext}",
                    "time_export.{ext}",
                    "time_{date}.{ext}",
                    "calendar_dim.{ext}",
                    "date_dim.{ext}",
                    "temporal_dim.{ext}",
                    "time_master.{ext}",
                    "staging_time.{ext}",
                    "time_snapshot.{ext}",
                ],
                "implicit": [
                    "calendar.{ext}",
                    "dates.{ext}",
                    "periods.{ext}",
                    "time_lookup.{ext}",
                    "fiscal_calendar.{ext}",
                    "time_mapping.{ext}",
                    "date_mapping.{ext}",
                    "timeline.{ext}",
                    "temporal_data.{ext}",
                    "schedule.{ext}",
                ],
                "ambiguous": [
                    "data.{ext}",
                    "lookup.{ext}",
                    "export.{ext}",
                    "master_file.{ext}",
                    "reference.{ext}",
                    "dump.{ext}",
                    "list.{ext}",
                    "records.{ext}",
                    "dataset.{ext}",
                    "output.{ext}",
                ],
                "abbreviations": [
                    "tm.{ext}",
                    "tmdim.{ext}",
                    "td.{ext}",
                    "time_dim.{ext}",
                    "tm_dim.{ext}",
                    "time_tbl.{ext}",
                    "time_ref.{ext}",
                    "tm_mstr.{ext}",
                    "tm_lkp.{ext}",
                    "tm_snp.{ext}",
                    "cal_dim.{ext}",
                    "dt_dim.{ext}",
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
                "geography": ["geo", "geog", "geogr", "location", "loc"],
                "distribution": ["dist", "distrib", "distr"],
                "merchandise": ["merch", "mdse", "mds"],
                "weighted": ["wtd", "wgtd", "wght"],
                "average": ["avg", "mean", "av"],
                "sales": ["sls", "sl"],
                "dollar": ["dlr", "dol", "$"],
                "unit": ["unt", "u"],
                "volume": ["vol", "v"],
                "total": ["tot", "ttl", "t"],
                "feature": ["feat", "ftr", "ft"],
                "display": ["disp", "dsp", "dply"],
                "price": ["prc", "pr"],
                "reduction": ["red", "reduct", "rdctn"],
                "incremental": ["incr", "incrmntl", "inc"],
                "base": ["bs", "baseline"],
                "weekly": ["wkly", "wk", "week"],
                "category": ["cat", "ctgry", "categ"],
                "discount": ["disc", "dsc", "dcnt"],
                "points": ["pts", "pt", "pnts"],
                "alternative": ["alt", "altrntv"],
                "adult": ["adlt", "ad"],
                "beverage": ["bev", "bvg", "drink"],
                "beer": ["br"],
                "vendor": ["vend", "vndr", "supplier", "supp"],
                "segment": ["seg", "sgmt", "sect"],
                "family": ["fam", "fmly"],
                "brand": ["brnd", "br"],
                "domestic": ["dom", "dmstc"],
                "import": ["imp", "imprt", "imported"],
                "versus": ["vs", "v"],
                "package": ["pkg", "pack", "pckg"],
                "size": ["sz", "sze"],
                "group": ["grp", "gp"],
                "type": ["typ", "tp"],
                "style": ["styl", "sty"],
                "macro": ["mac", "mcro"],
                "micro": ["mic", "mcro"],
                "mezzo": ["mez", "mzzo"],
                "count": ["cnt", "ct"],
                "time": ["tm", "t"],
                "thirteen": ["13", "13_digit"],
                "digit": ["dgt", "dig"],
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
                "geography": ["location", "region", "area", "territory", "place"],
                "sales": ["revenue", "turnover", "proceeds", "receipts"],
                "dollar": ["revenue", "monetary", "currency", "money"],
                "unit": ["quantity", "qty", "piece", "each"],
                "volume": ["capacity", "size", "amount", "bulk"],
                "distribution": ["allocation", "spread", "coverage"],
                "merchandise": ["promotion", "marketing", "advertising"],
                "feature": ["highlight", "promotion", "advertised"],
                "display": ["showcase", "exhibition", "presentation"],
                "price": ["cost", "rate", "charge", "fee"],
                "reduction": ["discount", "markdown", "decrease", "cut"],
                "incremental": ["additional", "extra", "marginal", "added"],
                "base": ["baseline", "foundation", "standard", "regular"],
                "average": ["mean", "typical", "standard"],
                "weighted": ["adjusted", "balanced", "modified"],
                "category": ["class", "type", "group", "segment"],
                "discount": ["reduction", "markdown", "rebate", "deduction"],
                "alternative": ["substitute", "option", "other"],
                "beverage": ["drink", "liquid", "refreshment"],
                "vendor": ["supplier", "provider", "manufacturer"],
                "segment": ["category", "division", "section"],
                "brand": ["label", "mark", "trademark"],
                "domestic": ["local", "national", "home"],
                "import": ["foreign", "international", "imported"],
                "package": ["container", "pack", "packaging"],
                "style": ["type", "kind", "variety"],
            },
            "prefixes": [
                "raw_",
                "src_",
                "orig_",
                "old_",
                "legacy_",
                "temp_",
                "new_",
                "stg_",
                "staging_",
                "dim_",
                "fact_",
                "agg_",
                "aggregate_",
                "sum_",
                "total_",
                "base_",
                "incr_",
                "incremental_",
            ],
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
                "_key",
                "_id",
                "_code",
                "_ref",
                "_amt",
                "_value",
                "_val",
                "_measure",
                "_metric",
                "_total",
                "_tot",
                "_sum",
                "_cnt",
                "_count",
                "_qty",
                "_quantity",
                "_desc",
                "_description",
                "_name",
                "_nm",
                "_flag",
                "_ind",
                "_indicator",
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

            # Explicit file names (30%)
            explicit_samples = self._generate_explicit_filenames(
                table, patterns["explicit"], int(samples_per_table * 0.3)
            )

            # Implicit file names (25%)
            implicit_samples = self._generate_implicit_filenames(
                table, patterns["implicit"], int(samples_per_table * 0.25)
            )

            # Abbreviations (20%)
            abbreviation_samples = self._generate_abbreviation_filenames(
                table, patterns["abbreviations"], int(samples_per_table * 0.2)
            )

            # Ambiguous file names (15%)
            ambiguous_samples = self._generate_ambiguous_filenames(
                table, patterns["ambiguous"], int(samples_per_table * 0.15)
            )

            # Noisy/misleading file names (10%)
            noisy_samples = self._generate_noisy_filenames(
                table, tables, int(samples_per_table * 0.1)
            )

            all_samples.extend(explicit_samples)
            all_samples.extend(implicit_samples)
            all_samples.extend(abbreviation_samples)
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

    def _generate_abbreviation_filenames(self, table, patterns, n_samples):
        """Generate file names using abbreviations and shortened forms"""
        samples = []
        extensions = ["csv", "xlsx", "txt", "dat"]
        dates = ["2024", "jan", "q1", "20240115", "2023", "dec"]

        # Common prefixes/suffixes for abbreviations
        prefixes = ["", "data_", "tbl_", "rpt_", "exp_", "imp_"]
        suffixes = ["", "_data", "_report", "_export", "_final", "_v1", "_backup"]
        separators = ["_", "-", ""]

        for _ in range(n_samples):
            pattern = random.choice(patterns)
            ext = random.choice(extensions)

            # Build filename with possible prefix/suffix
            prefix = random.choice(prefixes) if random.random() > 0.6 else ""
            suffix = random.choice(suffixes) if random.random() > 0.6 else ""
            separator = random.choice(separators)

            # Handle date placeholder if present
            if "{date}" in pattern:
                date_val = random.choice(dates)
                base_name = pattern.format(date=date_val)
            else:
                base_name = pattern

            # Construct full filename
            if prefix and separator:
                filename = f"{prefix}{separator}{base_name}"
            else:
                filename = f"{prefix}{base_name}"

            if suffix:
                filename = f"{filename}{suffix}"

            filename = f"{filename}.{ext}"

            # Apply case variations
            filename = self._apply_case_variation(filename)

            # Occasionally remove separators for more ambiguity
            if random.random() > 0.8:
                filename = filename.replace("_", "").replace("-", "")

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

        print("\nGenerating Final Two-Step Test dataset...")
        final_test = self.generate_final_test_dataset(n_samples=10000)
        final_test.to_csv(f"{output_dir}/final_two_step_test.csv", index=False)

        print(f"Final Test - Samples: {len(final_test)}")
        print(
            f"Final Test table distribution:\n{final_test['True Table Name'].value_counts()}"
        )

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

    def generate_final_test_dataset(self, n_samples=10000, test_seed=999):
        """
        Generate final two-step test dataset with both file name and column mapping.
        Uses a different random seed to ensure no overlap with train/val data.

        Args:
            n_samples: Total number of test samples to generate
            test_seed: Different random seed for test data generation

        Returns:
            DataFrame with columns: input_file_name, input_column_name,
                                    output_table_name, output_column_name
        """
        # Temporarily set different random seed for test data
        original_random_state = random.getstate()
        original_np_state = np.random.get_state()

        random.seed(test_seed)
        np.random.seed(test_seed)

        samples = []
        tables = list(self.domains.keys())
        samples_per_table = n_samples // len(tables)

        for table in tables:
            standard_columns = self.domains[table]["standard_columns"]
            patterns = self.file_patterns[table]

            # Get all filename patterns for this table
            all_patterns = (
                patterns["explicit"]
                + patterns["implicit"]
                + patterns["abbreviations"]
                + patterns["ambiguous"]
            )

            # Generate samples for each standard column in this table
            samples_per_column = max(1, samples_per_table // len(standard_columns))

            for standard_col in standard_columns:
                for _ in range(samples_per_column):
                    # Generate a filename for this table
                    pattern = random.choice(all_patterns)
                    ext = random.choice(["csv", "xlsx", "parquet", "tsv"])

                    # Handle date/version placeholders
                    filename = pattern.format(
                        ext=ext,
                        date=(
                            random.choice(["2024", "jan_2024", "q1_2024"])
                            if "{date}" in pattern
                            else ""
                        ),
                        version=(
                            random.choice(["v1", "final"])
                            if "{version}" in pattern
                            else ""
                        ),
                    )

                    # Apply case variation
                    filename = self._apply_case_variation(filename)

                    # Generate a transformed version of the standard column
                    raw_column = self._transform_column_name(standard_col)

                    samples.append(
                        {
                            "Input File Name": filename,
                            "Input Column Name": raw_column,
                            "True Table Name": table,
                            "True Column Name": standard_col,
                        }
                    )

        # Shuffle the samples
        random.shuffle(samples)

        # Restore original random state
        random.setstate(original_random_state)
        np.random.set_state(original_np_state)

        return pd.DataFrame(samples)


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
