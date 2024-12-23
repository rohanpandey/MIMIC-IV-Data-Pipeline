#!/usr/bin/env python
# coding: utf-8

import polars as pl
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "./../..")


def read_icd_mapping(map_path: str) -> pl.DataFrame:
    """Reads in mapping table for converting ICD9 to ICD10 codes"""

    mapping = pl.read_csv(map_path, has_header=True, separator="\t")
    mapping = mapping.with_columns(pl.col("diagnosis_description").str.to_lowercase())
    return mapping


def get_diagnosis_icd(module_path: str) -> pl.DataFrame:
    """Reads in diagnosis_icd table"""

    return pl.read_csv(module_path + "/hosp/diagnoses_icd.csv.gz", has_header=True)


def standardize_icd(
    mapping: pl.DataFrame,
    diag: pl.DataFrame,
    map_code_col="diagnosis_code",
    root=True,
) -> pl.DataFrame:
    """Takes an ICD9 -> ICD10 mapping table and a diagnosis dataframe;
    adds column with converted ICD10 codes"""

    # Prepare the mapping DataFrame
    if root:
        mapping = mapping.with_columns(
            pl.col(map_code_col).str.slice(0, 3).alias("diagnosis_code_root")
        )
    else:
        mapping = mapping.with_columns(
            pl.col(map_code_col).alias("diagnosis_code_root")
        )

    # Prepare the diag DataFrame
    diag = diag.with_columns(
        pl.when(pl.col("icd_version") == 9)
        .then(pl.col("icd_code").str.slice(0, 3) if root else pl.col("icd_code"))
        .otherwise(None)
        .alias("icd_code_root")
    )

    # Initialize the 'root_icd10_convert' column
    diag = diag.with_columns(pl.col("icd_code").alias("root_icd10_convert"))

    # Join to map ICD9 to ICD10 codes
    diag = diag.join(
        mapping.select(["diagnosis_code_root", "icd10cm"]),
        left_on="icd_code_root",
        right_on="diagnosis_code_root",
        how="left",
    )

    # Update 'root_icd10_convert' where mapping exists
    diag = diag.with_columns(
        pl.when((pl.col("icd_version") == 9) & (pl.col("icd10cm").is_not_null()))
        .then(pl.col("icd10cm"))
        .otherwise(pl.col("root_icd10_convert"))
        .alias("root_icd10_convert")
    )

    # Create the 'root' column from the first three characters
    diag = diag.with_columns(pl.col("root_icd10_convert").str.slice(0, 3).alias("root"))

    return diag


def preproc_icd_module(
    h_ids, module_path: str, ICD10_code: str, icd_map_path: str
) -> pl.DataFrame:
    """Processes the diagnosis module and returns positive IDs"""

    diag = get_diagnosis_icd(module_path)
    icd_map = read_icd_mapping(icd_map_path)

    diag = standardize_icd(icd_map, diag, root=True)

    # Filter out records with null 'root' and those matching the ICD10 code
    diag = diag.drop_nulls(subset=["root"])
    pos_ids = (
        diag.filter(pl.col("root").str.contains(ICD10_code)).select("hadm_id").unique()
    )

    return pos_ids


def extract_diag_cohort(
    h_ids,
    label: str,
    module_path,
    icd_map_path="./utils/mappings/ICD9_to_ICD10_mapping.txt",
) -> pl.DataFrame:
    """Extracts the diagnosis cohort based on ICD codes"""

    cohort = preproc_icd_module(h_ids, module_path, label, icd_map_path)
    return cohort
