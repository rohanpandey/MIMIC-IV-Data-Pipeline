import csv
import numpy as np
import polars as pl
import pandas as pd
import sys, os
import re
import ast
import datetime as dt
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer


########################## GENERAL ##########################
def dataframe_from_csv(path, compression="gzip"):
    return pl.read_csv(path)


def read_admissions_table(mimic4_path):
    path = os.path.join(mimic4_path, "core/admissions.csv.gz")
    admits = pl.read_csv(path, try_parse_dates=True)
    admits = admits.select(
        ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "ethnicity"]
    )
    return admits


def read_patients_table(mimic4_path):
    path = os.path.join(mimic4_path, "core/patients.csv.gz")
    pats = pl.read_csv(path, try_parse_dates=True)
    pats = pats.select(
        [
            "subject_id",
            "gender",
            "dod",
            "anchor_age",
            "anchor_year",
            "anchor_year_group",
        ]
    )
    pats = pats.with_column((pl.col("anchor_year") - pl.col("anchor_age")).alias("yob"))
    return pats


########################## DIAGNOSES ##########################
def read_diagnoses_icd_table(mimic4_path):
    path = os.path.join(mimic4_path, "hosp/diagnoses_icd.csv.gz")
    diag = pl.read_csv(path)
    return diag


def read_d_icd_diagnoses_table(mimic4_path):
    path = os.path.join(mimic4_path, "hosp/d_icd_diagnoses.csv.gz")
    d_icd = pl.read_csv(path)
    d_icd = d_icd.select(["icd_code", "long_title"])
    return d_icd


def read_diagnoses(mimic4_path):
    diag_icd = read_diagnoses_icd_table(mimic4_path)
    d_icd = read_d_icd_diagnoses_table(mimic4_path)
    diag = diag_icd.join(d_icd, on="icd_code", how="inner")
    return diag


def standardize_icd(mapping: pl.DataFrame, df: pl.DataFrame, root=False):
    """Takes an ICD9 -> ICD10 mapping table and a diagnosis dataframe; adds column with converted ICD10 column"""

    col_name = "icd10_convert"
    if root:
        col_name = "root_" + col_name

    if root:
        df = df.with_column(
            pl.when(pl.col("icd_version") == 9)
            .then(pl.col("icd_code").str.slice(0, 3))
            .alias("icd9_code")
        )
    else:
        df = df.with_column(
            pl.when(pl.col("icd_version") == 9)
            .then(pl.col("icd_code"))
            .alias("icd9_code")
        )

    df = df.join(mapping, left_on="icd9_code", right_on="diagnosis_code", how="left")

    df = df.with_column(
        pl.when(pl.col("icd_version") == 9)
        .then(
            pl.when(pl.col("icd10cm").is_null())
            .then(pl.col("icd_code"))
            .otherwise(pl.col("icd10cm"))
        )
        .otherwise(pl.col("icd_code"))
        .alias(col_name)
    )

    df = df.drop(["icd9_code", "diagnosis_code", "icd10cm", "diagnosis_description"])
    return df


########################## PROCEDURES ##########################
def read_procedures_icd_table(mimic4_path):
    path = os.path.join(mimic4_path, "hosp/procedures_icd.csv.gz")
    proc = pl.read_csv(path)
    return proc


def read_d_icd_procedures_table(mimic4_path):
    path = os.path.join(mimic4_path, "hosp/d_icd_procedures.csv.gz")
    p_icd = pl.read_csv(path)
    p_icd = p_icd.select(["icd_code", "long_title"])
    return p_icd


def read_procedures(mimic4_path):
    proc_icd = read_procedures_icd_table(mimic4_path)
    p_icd = read_d_icd_procedures_table(mimic4_path)
    proc = proc_icd.join(p_icd, on="icd_code", how="inner")
    return proc


########################## MAPPING ##########################
def read_icd_mapping(map_path):
    mapping = pl.read_csv(map_path, separator="\t")
    mapping = mapping.with_column(pl.col("diagnosis_description").str.to_lowercase())
    return mapping


########################## PREPROCESSING ##########################


def preproc_meds(module_path: str, adm_cohort_path: str) -> pl.DataFrame:

    adm = pl.read_csv(
        adm_cohort_path, columns=["hadm_id", "stay_id", "intime"], try_parse_dates=True
    )
    med = pl.read_csv(
        module_path,
        columns=[
            "subject_id",
            "stay_id",
            "itemid",
            "starttime",
            "endtime",
            "rate",
            "amount",
            "orderid",
        ],
        parse_dates=True,
    )
    med = med.join(adm, on="stay_id", how="inner")
    med = med.with_columns(
        [
            (pl.col("starttime") - pl.col("intime")).alias("start_hours_from_admit"),
            (pl.col("endtime") - pl.col("intime")).alias("stop_hours_from_admit"),
        ]
    )
    med = med.drop_nulls()
    print("# of unique type of drug: ", med["itemid"].n_unique())
    print("# Admissions:  ", med["stay_id"].n_unique())
    print("# Total rows", med.shape[0])
    return med


def preproc_proc(
    dataset_path: str, cohort_path: str, time_col: str, dtypes: dict, usecols: list
) -> pl.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""

    def merge_module_cohort() -> pl.DataFrame:
        """Gets the initial module data with patients anchor year data and only the year of the charttime"""
        module = pl.read_csv(
            dataset_path,
            columns=usecols,
            dtypes=dtypes,
            try_parse_dates=True,
        ).unique()
        cohort = pl.read_csv(cohort_path, try_parse_dates=True)
        return module.join(
            cohort.select(["subject_id", "hadm_id", "stay_id", "intime", "outtime"]),
            on="stay_id",
            how="inner",
        )

    df_cohort = merge_module_cohort()
    df_cohort = df_cohort.with_column(
        (pl.col(time_col) - pl.col("intime")).alias("event_time_from_admit")
    )
    df_cohort = df_cohort.drop_nulls()
    print("# Unique Events:  ", df_cohort["itemid"].drop_nulls().n_unique())
    print("# Admissions:  ", df_cohort["stay_id"].n_unique())
    print("Total rows", df_cohort.shape[0])
    return df_cohort


def preproc_out(
    dataset_path: str, cohort_path: str, time_col: str, dtypes: dict, usecols: list
) -> pl.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""

    def merge_module_cohort() -> pl.DataFrame:
        module = pl.read_csv(
            dataset_path,
            columns=usecols,
            dtypes=dtypes,
            try_parse_dates=True,
        ).unique()
        cohort = pl.read_csv(cohort_path, try_parse_dates=True)
        return module.join(
            cohort.select(["stay_id", "intime", "outtime"]), on="stay_id", how="inner"
        )

    df_cohort = merge_module_cohort()
    df_cohort = df_cohort.with_column(
        (pl.col(time_col) - pl.col("intime")).alias("event_time_from_admit")
    )
    df_cohort = df_cohort.drop_nulls()
    print("# Unique Events:  ", df_cohort["itemid"].n_unique())
    print("# Admissions:  ", df_cohort["stay_id"].n_unique())
    print("Total rows", df_cohort.shape[0])
    return df_cohort


def preproc_chart(
    dataset_path: str, cohort_path: str, time_col: str, dtypes: dict, usecols: list
) -> pl.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""

    # using scan_csv, polars will automatically try to use streaming, which is more memory efficient
    # every operation in this function is lazy, so no data is loaded into memory until the final collect

    cohort = pl.scan_csv(cohort_path, try_parse_dates=True)

    query = (
        pl.scan_csv(dataset_path, try_parse_dates=True)
        .select(usecols)
        .drop_nulls(subset=["valuenum"])
        .join(cohort.select(["stay_id", "intime"]), on="stay_id", how="inner")
        .with_columns(
            (pl.col(time_col) - pl.col("intime")).alias("event_time_from_admit")
        )
        .drop_nulls()
        .unique()
    )

    df_cohort = query.collect()

    print("# Unique Events:  ", df_cohort["itemid"].n_unique())
    print("# Admissions:  ", df_cohort["stay_id"].n_unique())
    print("Total rows", df_cohort.shape[0])
    return df_cohort


def preproc_icd_module(
    module_path: str,
    adm_cohort_path: str,
    icd_map_path=None,
    map_code_colname=None,
    only_icd10=True,
) -> pl.DataFrame:
    """Takes an module dataset with ICD codes and puts it in long_format, optionally mapping ICD-codes by a mapping table path"""

    def get_module_cohort(module_path: str, cohort_path: str):
        module = pl.read_csv(module_path)
        adm_cohort = pl.read_csv(adm_cohort_path)
        return module.join(
            adm_cohort.select(["hadm_id", "stay_id", "label"]),
            on="hadm_id",
            how="inner",
        )

    def standardize_icd(mapping, df, root=False):
        """Takes an ICD9 -> ICD10 mapping table and a diagnosis dataframe; adds column with converted ICD10 column"""

        col_name = "icd10_convert"
        if root:
            col_name = "root_" + col_name

        if root:
            df = df.with_column(
                pl.when(pl.col("icd_version") == 9)
                .then(pl.col("icd_code").str.slice(0, 3))
                .alias("icd9_code")
            )
        else:
            df = df.with_column(
                pl.when(pl.col("icd_version") == 9)
                .then(pl.col("icd_code"))
                .alias("icd9_code")
            )

        df = df.join(
            mapping, left_on="icd9_code", right_on=map_code_colname, how="left"
        )

        df = df.with_column(
            pl.when(pl.col("icd_version") == 9)
            .then(
                pl.when(pl.col("icd10cm").is_null())
                .then(pl.col("icd_code"))
                .otherwise(pl.col("icd10cm"))
            )
            .otherwise(pl.col("icd_code"))
            .alias(col_name)
        )

        if only_icd10:
            df = df.with_column(pl.col(col_name).str.slice(0, 3).alias("root"))

        df = df.drop(
            ["icd9_code", map_code_colname, "icd10cm", "diagnosis_description"]
        )
        return df

    module = get_module_cohort(module_path, adm_cohort_path)
    if icd_map_path:
        icd_map = read_icd_mapping(icd_map_path)
        module = standardize_icd(icd_map, module, root=True)
        print(
            "# unique ICD-9 codes",
            module.filter(pl.col("icd_version") == 9)["icd_code"].n_unique(),
        )
        print(
            "# unique ICD-10 codes",
            module.filter(pl.col("icd_version") == 10)["icd_code"].n_unique(),
        )
        print(
            "# unique ICD-10 codes (After converting ICD-9 to ICD-10)",
            module["root_icd10_convert"].n_unique(),
        )
        print(
            "# unique ICD-10 codes (After clinical grouping ICD-10 codes)",
            module["root"].n_unique(),
        )
        print("# Admissions:  ", module["stay_id"].n_unique())
        print("Total rows", module.shape[0])
    return module


def pivot_cohort(
    df: pl.DataFrame,
    prefix: str,
    target_col: str,
    values="values",
    use_mlb=False,
    ohe=True,
    max_features=None,
):
    """Pivots long_format data into a multiindex array:
                                        || feature 1 || ... || feature n ||
    || subject_id || label || timedelta ||
    """
    if use_mlb:
        mlb = MultiLabelBinarizer()
        df_pandas = df.to_pandas()
        # pandas still gets used here because of the literal_eval function
        output = mlb.fit_transform(df_pandas[target_col].apply(ast.literal_eval))
        output_df = pl.from_pandas(pd.DataFrame(output, columns=mlb.classes_))
        if max_features:
            top_features = output_df.sum().sort("values", reverse=True)[:max_features][
                "feature"
            ]
            output_df = output_df.select(top_features)
        df_pandas = pd.concat(
            [
                df_pandas[["subject_id", "label", "timedelta"]].reset_index(drop=True),
                output_df.to_pandas(),
            ],
            axis=1,
        )
        pivot_df = df_pandas.pivot_table(
            index=["subject_id", "label", "timedelta"],
            values=df_pandas.columns[3:],
            aggfunc=np.max,
        )
        pivot_df.columns = [prefix + str(i) for i in pivot_df.columns]
        return pl.from_pandas(pivot_df)
    else:
        if max_features:
            top_features = (
                df.select(target_col)
                .unique()
                .to_series()
                .value_counts()
                .head(max_features)
                .index
            )
            df = df.filter(pl.col(target_col).is_in(top_features))
        if ohe:
            df = df.with_column(pl.lit(1).alias("values"))
            pivot_df = df.pivot(
                values="values",
                index=["subject_id", "label", "timedelta"],
                columns=target_col,
                aggregate_function="max",
            )
        else:
            pivot_df = df.pivot(
                values=values,
                index=["subject_id", "label", "timedelta"],
                columns=target_col,
                aggregate_function="mean",
            )
        pivot_df = pivot_df.select(
            [pl.col(col).alias(prefix + str(col)) for col in pivot_df.columns]
        )
        return pivot_df
