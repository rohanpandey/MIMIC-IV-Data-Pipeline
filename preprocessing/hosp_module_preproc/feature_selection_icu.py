import os
import importlib
import gzip

# Assuming the utility modules have been updated to use Polars
import utils.icu_preprocess_util
from utils.icu_preprocess_util import *

importlib.reload(utils.icu_preprocess_util)
import utils.icu_preprocess_util
from utils.icu_preprocess_util import *

import utils.outlier_removal
from utils.outlier_removal import *

importlib.reload(utils.outlier_removal)
import utils.outlier_removal
from utils.outlier_removal import *

import utils.uom_conversion
from utils.uom_conversion import *
import polars as pl

if not os.path.exists("./data/features"):
    os.makedirs("./data/features")
if not os.path.exists("./data/features/chartevents"):
    os.makedirs("./data/features/chartevents")


def feature_icu(
    cohort_output,
    version_path,
    diag_flag=True,
    out_flag=True,
    chart_flag=True,
    proc_flag=True,
    med_flag=True,
    mimiciv_path="",
):
    if diag_flag:
        print("[EXTRACTING DIAGNOSIS DATA]")
        diag = preproc_icd_module(
            mimiciv_path + version_path + "/hosp/diagnoses_icd.csv.gz",
            "./data/cohort/" + cohort_output + ".csv",
            "./utils/mappings/ICD9_to_ICD10_mapping.txt",
            map_code_colname="diagnosis_code",
        )
        diag = diag.select(
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "icd_code",
                "root_icd10_convert",
                "root",
            ]
        )
        with gzip.open("./data/features/preproc_diag_icu.csv.gz", "wt") as f:
            diag.write_csv(f)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if out_flag:
        print("[EXTRACTING OUTPUT EVENTS DATA]")
        out = preproc_out(
            mimiciv_path + version_path + "/icu/outputevents.csv.gz",
            "./data/cohort/" + cohort_output + ".csv",
            "charttime",
            dtypes=None,
            usecols=None,
        )
        out = out.select(
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "itemid",
                "charttime",
                "intime",
                "event_time_from_admit",
            ]
        )
        with gzip.open("./data/features/preproc_out_icu.csv.gz", "wt") as f:
            out.write_csv(f)
        print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")

    if chart_flag:
        print("[EXTRACTING CHART EVENTS DATA]")
        chart = preproc_chart(
            mimiciv_path + version_path + "/icu/chartevents.csv.gz",
            "./data/cohort/" + cohort_output + ".csv",
            "charttime",
            dtypes=None,
            usecols=["stay_id", "charttime", "itemid", "valuenum", "valueuom"],
        )
        # COMMENTED BY ROHAN
        # chart = drop_wrong_uom(chart, 0.95)
        chart = chart.select(
            ["stay_id", "itemid", "event_time_from_admit", "valuenum"]
        )  # convert event_time_from_admit to string as days, hours, minutes, etc
        chart = chart.with_columns(
            pl.col("event_time_from_admit")
            .dt.to_string(format="polars")
            .alias("event_time_from_admit_str")
        )
        chart = chart.select(
            ["stay_id", "itemid", "event_time_from_admit_str", "valuenum"]
        )

        with gzip.open("./data/features/preproc_chart_icu.csv.gz", "wt") as f:
            chart.write_csv(f)
        print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")

    if proc_flag:
        print("[EXTRACTING PROCEDURES DATA]")
        proc = preproc_proc(
            mimiciv_path + version_path + "/icu/procedureevents.csv.gz",
            "./data/cohort/" + cohort_output + ".csv",
            "starttime",
            dtypes=None,
            usecols=["stay_id", "starttime", "itemid"],
        )
        proc = proc.select(
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "itemid",
                "starttime",
                "intime",
                "event_time_from_admit",
            ]
        )
        with gzip.open("./data/features/preproc_proc_icu.csv.gz", "wt") as f:
            proc.write_csv(f)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")

    if med_flag:
        print("[EXTRACTING MEDICATIONS DATA]")
        med = preproc_meds(
            mimiciv_path + version_path + "/icu/inputevents.csv.gz",
            "./data/cohort/" + cohort_output + ".csv",
        )
        med = med.select(
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "itemid",
                "starttime",
                "endtime",
                "start_hours_from_admit",
                "stop_hours_from_admit",
                "rate",
                "amount",
                "orderid",
            ]
        )
        with gzip.open("./data/features/preproc_med_icu.csv.gz", "wt") as f:
            med.write_csv(f)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")


def preprocess_features_icu(
    cohort_output,
    diag_flag,
    group_diag,
    chart_flag,
    clean_chart,
    impute_outlier_chart,
    thresh,
    left_thresh,
):
    if diag_flag:
        print("[PROCESSING DIAGNOSIS DATA]")
        diag = pl.read_csv("./data/features/preproc_diag_icu.csv.gz")
        if group_diag == "Keep both ICD-9 and ICD-10 codes":
            diag = diag.with_column(pl.col("icd_code").alias("new_icd_code"))
        if group_diag == "Convert ICD-9 to ICD-10 codes":
            diag = diag.with_column(pl.col("root_icd10_convert").alias("new_icd_code"))
        if group_diag == "Convert ICD-9 to ICD-10 and group ICD-10 codes":
            diag = diag.with_column(pl.col("root").alias("new_icd_code"))

        diag = diag.select(
            ["subject_id", "hadm_id", "stay_id", "new_icd_code"]
        ).drop_nulls()
        print("Total number of rows", diag.shape[0])
        with gzip.open("./data/features/preproc_diag_icu.csv.gz", "wt") as f:
            diag.write_csv(f)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if chart_flag:
        if clean_chart:
            print("[PROCESSING CHART EVENTS DATA]")
            chart = pl.read_csv("./data/features/preproc_chart_icu.csv.gz")
            chart = outlier_imputation(
                chart, "itemid", "valuenum", thresh, left_thresh, impute_outlier_chart
            )
            # Uncomment and adjust if necessary
            # for i in [227441, 229357, 229358, 229360]:
            #     try:
            #         maj = chart.filter(pl.col('itemid') == i)['valueuom'].mode()
            #         chart = chart.filter(~((pl.col('itemid') == i) & (pl.col('valueuom') == maj)))
            #     except IndexError:
            #         print(f"{i} not found")
            print("Total number of rows", chart.shape[0])
            with gzip.open("./data/features/preproc_chart_icu.csv.gz", "wt") as f:
                chart.write_csv(f)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")


def generate_summary_icu(diag_flag, proc_flag, med_flag, out_flag, chart_flag):
    print("[GENERATING FEATURE SUMMARY]")
    if diag_flag:
        diag = pl.read_csv("./data/features/preproc_diag_icu.csv.gz")
        freq = diag.groupby(["stay_id", "new_icd_code"]).agg(mean_frequency=pl.count())
        freq = freq.groupby("new_icd_code").agg(
            mean_frequency=pl.col("mean_frequency").mean()
        )
        total = diag.groupby("new_icd_code").agg(total_count=pl.count())
        summary = freq.join(total, on="new_icd_code", how="right")
        summary = summary.fill_null(0)
        summary.write_csv("./data/summary/diag_summary.csv")
        summary.select("new_icd_code").write_csv("./data/summary/diag_features.csv")

    if med_flag:
        med = pl.read_csv("./data/features/preproc_med_icu.csv.gz")
        freq = med.groupby(["stay_id", "itemid"]).agg(mean_frequency=pl.count())
        freq = freq.groupby("itemid").agg(
            mean_frequency=pl.col("mean_frequency").mean()
        )
        missing = (
            med.filter(pl.col("amount") == 0)
            .groupby("itemid")
            .agg(missing_count=pl.count())
        )
        total = med.groupby("itemid").agg(total_count=pl.count())
        summary = missing.join(total, on="itemid", how="outer")
        summary = freq.join(summary, on="itemid", how="right")
        summary = summary.fill_null(0)
        summary.write_csv("./data/summary/med_summary.csv")
        summary.select("itemid").write_csv("./data/summary/med_features.csv")

    if proc_flag:
        proc = pl.read_csv("./data/features/preproc_proc_icu.csv.gz")
        freq = proc.groupby(["stay_id", "itemid"]).agg(mean_frequency=pl.count())
        freq = freq.groupby("itemid").agg(
            mean_frequency=pl.col("mean_frequency").mean()
        )
        total = proc.groupby("itemid").agg(total_count=pl.count())
        summary = freq.join(total, on="itemid", how="right")
        summary = summary.fill_null(0)
        summary.write_csv("./data/summary/proc_summary.csv")
        summary.select("itemid").write_csv("./data/summary/proc_features.csv")

    if out_flag:
        out = pl.read_csv("./data/features/preproc_out_icu.csv.gz")
        freq = out.groupby(["stay_id", "itemid"]).agg(mean_frequency=pl.count())
        freq = freq.groupby("itemid").agg(
            mean_frequency=pl.col("mean_frequency").mean()
        )
        total = out.groupby("itemid").agg(total_count=pl.count())
        summary = freq.join(total, on="itemid", how="right")
        summary = summary.fill_null(0)
        summary.write_csv("./data/summary/out_summary.csv")
        summary.select("itemid").write_csv("./data/summary/out_features.csv")

    if chart_flag:
        chart = pl.read_csv("./data/features/preproc_chart_icu.csv.gz")
        freq = chart.groupby(["stay_id", "itemid"]).agg(mean_frequency=pl.count())
        freq = freq.groupby("itemid").agg(
            mean_frequency=pl.col("mean_frequency").mean()
        )
        missing = (
            chart.filter(pl.col("valuenum") == 0)
            .groupby("itemid")
            .agg(missing_count=pl.count())
        )
        total = chart.groupby("itemid").agg(total_count=pl.count())
        summary = missing.join(total, on="itemid", how="outer")
        summary = freq.join(summary, on="itemid", how="right")
        summary = summary.fill_null(0)
        summary.write_csv("./data/summary/chart_summary.csv")
        summary.select("itemid").write_csv("./data/summary/chart_features.csv")
    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")


def features_selection_icu(
    cohort_output,
    diag_flag,
    proc_flag,
    med_flag,
    out_flag,
    chart_flag,
    group_diag,
    group_med,
    group_proc,
    group_out,
    group_chart,
):
    if diag_flag and group_diag:
        print("[FEATURE SELECTION DIAGNOSIS DATA]")
        diag = pl.read_csv("./data/features/preproc_diag_icu.csv.gz")
        features = pl.read_csv("./data/summary/diag_features.csv")
        diag = diag.filter(pl.col("new_icd_code").is_in(features["new_icd_code"]))
        print("Total number of rows", diag.shape[0])
        with gzip.open("./data/features/preproc_diag_icu.csv.gz", "wt") as f:
            diag.write_csv(f)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if med_flag and group_med:
        print("[FEATURE SELECTION MEDICATIONS DATA]")
        med = pl.read_csv("./data/features/preproc_med_icu.csv.gz")
        features = pl.read_csv("./data/summary/med_features.csv")
        med = med.filter(pl.col("itemid").is_in(features["itemid"]))
        print("Total number of rows", med.shape[0])
        with gzip.open("./data/features/preproc_med_icu.csv.gz", "wt") as f:
            med.write_csv(f)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

    if proc_flag and group_proc:
        print("[FEATURE SELECTION PROCEDURES DATA]")
        proc = pl.read_csv("./data/features/preproc_proc_icu.csv.gz")
        features = pl.read_csv("./data/summary/proc_features.csv")
        proc = proc.filter(pl.col("itemid").is_in(features["itemid"]))
        print("Total number of rows", proc.shape[0])
        with gzip.open("./data/features/preproc_proc_icu.csv.gz", "wt") as f:
            proc.write_csv(f)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")

    if out_flag and group_out:
        print("[FEATURE SELECTION OUTPUT EVENTS DATA]")
        out = pl.read_csv("./data/features/preproc_out_icu.csv.gz")
        features = pl.read_csv("./data/summary/out_features.csv")
        out = out.filter(pl.col("itemid").is_in(features["itemid"]))
        print("Total number of rows", out.shape[0])
        with gzip.open("./data/features/preproc_out_icu.csv.gz", "wt") as f:
            out.write_csv(f)
        print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")

    if chart_flag and group_chart:
        print("[FEATURE SELECTION CHART EVENTS DATA]")
        chart = pl.read_csv("./data/features/preproc_chart_icu.csv.gz")
        features = pl.read_csv("./data/summary/chart_features.csv")
        chart = chart.filter(pl.col("itemid").is_in(features["itemid"]))
        print("Total number of rows", chart.shape[0])
        with gzip.open("./data/features/preproc_chart_icu.csv.gz", "wt") as f:
            chart.write_csv(f)
        print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
