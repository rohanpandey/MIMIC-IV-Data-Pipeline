import datetime
import os
import sys
import polars as pl
import importlib
import disease_cohort

importlib.reload(disease_cohort)
import disease_cohort

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "./../..")
os.makedirs("./data/cohort", exist_ok=True)


def get_visit_pts(
    mimic4_path,
    group_col,
    visit_col,
    admit_col,
    disch_col,
    adm_visit_col,
    use_admn,
    disease_label,
    use_ICU,
):
    """
    Combines the MIMIC-IV core/patients table information with either the icu/icustays or core/admissions data.
    """
    file_path = "icu/icustays.csv.gz" if use_ICU else "hosp/admissions.csv.gz"
    visit = pl.read_csv(
        mimic4_path + file_path,
        columns=None,
        try_parse_dates=True,
        dtypes={admit_col: pl.Datetime, disch_col: pl.Datetime},
    )

    if not use_ICU:
        visit = visit.with_columns(
            (pl.col(disch_col) - pl.col(admit_col)).dt.days().alias("los")
        )
        if use_admn:
            # remove hospitalizations with a death; impossible for readmission for such visits
            visit = visit.filter(pl.col("hospital_expire_flag") == 0)

    if use_admn:
        pts = pl.read_csv(
            mimic4_path + "hosp/patients.csv.gz",
            columns=["subject_id", "dod"],
            dtypes={"dod": pl.Datetime},
        )
        visit = visit.join(pts, on="subject_id", how="inner")
        visit = visit.filter(
            pl.col("dod").is_null() | (pl.col("dod") >= pl.col(disch_col))
        )

    if disease_label:
        hids = disease_cohort.extract_diag_cohort(
            visit.select("hadm_id"), disease_label, mimic4_path
        )
        visit = visit.filter(pl.col("hadm_id").is_in(hids["hadm_id"].to_list()))
        print(f"[ READMISSION DUE TO {disease_label} ]")

    pts = pl.read_csv(
        mimic4_path + "hosp/patients.csv.gz",
        columns=[
            group_col,
            "anchor_year",
            "anchor_age",
            "anchor_year_group",
            "dod",
            "gender",
        ],
    )
    pts = pts.with_columns(
        [
            (pl.col("anchor_year") - pl.col("anchor_age")).alias("yob"),
            (
                pl.col("anchor_year")
                + (2019 - pl.col("anchor_year_group").str.slice(-4).cast(pl.Int32))
            ).alias("min_valid_year"),
        ]
    )

    if use_ICU:
        selected_visit_cols = [
            group_col,
            visit_col,
            adm_visit_col,
            admit_col,
            disch_col,
            "los",
        ]
    else:
        selected_visit_cols = [group_col, visit_col, admit_col, disch_col, "los"]

    visit = visit.select(selected_visit_cols)
    visit_pts = visit.join(pts, on=group_col, how="inner")

    print(visit_pts.head())

    visit_pts = visit_pts.with_columns(pl.col("anchor_age").alias("Age"))
    visit_pts = visit_pts.filter(pl.col("Age") >= 18)

    eth = pl.read_csv(
        mimic4_path + "hosp/admissions.csv.gz", columns=["hadm_id", "insurance", "race"]
    )
    visit_pts = visit_pts.join(eth, on="hadm_id", how="inner")
    if not use_ICU:
        visit_pts = visit_pts.drop_nulls(subset=["min_valid_year"])
    return visit_pts


def partition_by_los(df, los, group_col, admit_col, disch_col):
    cohort = df.drop_nulls(subset=[admit_col, disch_col, "los"])
    pos_cohort = cohort.filter(pl.col("los") > los).fill_null(0)
    neg_cohort = cohort.filter(pl.col("los") <= los).fill_null(0)

    pos_cohort = pos_cohort.with_columns(pl.lit(1).alias("label"))
    neg_cohort = neg_cohort.with_columns(pl.lit(0).alias("label"))
    print("[ LOS LABELS FINISHED ]")
    return pl.concat([pos_cohort, neg_cohort]).sort([group_col, admit_col])


def partition_by_readmit(df, gap, group_col, admit_col, disch_col, valid_col):
    df = df.sort([group_col, admit_col])
    df = df.with_columns(
        [
            pl.col(admit_col).shift(-1).over(group_col).alias("next_admit"),
            pl.col(disch_col).cast(pl.Datetime).alias("current_disch"),
        ]
    )
    df = df.with_columns(
        (pl.col("next_admit") - pl.col("current_disch")).alias("gap_to_next_admit")
    )
    df = df.with_columns(
        pl.when(
            (pl.col("gap_to_next_admit") <= gap)
            & pl.col("gap_to_next_admit").is_not_null()
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("label")
    )
    case = df.filter(pl.col("label") == 1)
    ctrl = df.filter(pl.col("label") == 0)
    print("[ READMISSION LABELS FINISHED ]")
    return case, ctrl


def partition_by_mort(df, group_col, admit_col, disch_col, death_col):
    cohort = df.filter(
        pl.col(admit_col).is_not_null() & pl.col(disch_col).is_not_null()
    )
    cohort = cohort.with_columns(pl.lit(0).alias("label"))
    pos_cohort = cohort.filter(pl.col(death_col).is_not_null()).fill_null(0)
    neg_cohort = cohort.filter(pl.col(death_col).is_null()).fill_null(0)
    pos_cohort = pos_cohort.with_columns(pl.col(death_col).str.to_datetime("%Y-%m-%d"))
    # both dataframes need same datatypes for concatenation
    neg_cohort = neg_cohort.with_columns(
        pl.col(death_col).str.to_datetime("%Y-%m-%d", strict=False)
    )

    pos_cohort = pos_cohort.with_columns(
        pl.when(
            (pl.col(death_col) >= pl.col(admit_col))
            & (pl.col(death_col) <= pl.col(disch_col))
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("label")
    )
    cohort = pl.concat([pos_cohort, neg_cohort]).sort([group_col, admit_col])
    print("[ MORTALITY LABELS FINISHED ]")
    return cohort


def get_case_ctrls(
    df,
    gap,
    group_col,
    visit_col,
    admit_col,
    disch_col,
    valid_col,
    death_col,
    use_mort=False,
    use_admn=False,
    use_los=False,
):
    if use_mort:
        return (
            partition_by_mort(df, group_col, admit_col, disch_col, death_col),
            pl.DataFrame(),
        )
    elif use_admn:
        gap_td = datetime.timedelta(days=gap)
        case, ctrl = partition_by_readmit(
            df, gap_td, group_col, admit_col, disch_col, valid_col
        )
        return pl.concat([case, ctrl]), pl.DataFrame()
    elif use_los:
        return (
            partition_by_los(df, gap, group_col, admit_col, disch_col),
            pl.DataFrame(),
        )


def extract_data(
    use_ICU,
    label,
    time,
    icd_code,
    root_dir,
    disease_label,
    cohort_output=None,
    summary_output=None,
):
    """
    Extracts cohort data and summary from MIMIC-IV data based on provided parameters.
    """
    cohort_output = (
        cohort_output
        or f"cohort_{use_ICU.lower()}_{label.lower().replace(' ', '_')}_{time}_{disease_label}"
    )
    summary_output = (
        summary_output
        or f"summary_{use_ICU.lower()}_{label.lower().replace(' ', '_')}_{time}_{disease_label}"
    )

    print(
        f"EXTRACTING FOR: | {use_ICU.upper()} | {label.upper()} {'DUE TO ' + disease_label.upper() if disease_label else ''} | {'ADMITTED DUE TO ' + icd_code.upper() if icd_code != 'No Disease Filter' else ''} | {time} |"
    )
    ICU = use_ICU
    use_mort = label == "Mortality"
    use_admn = label == "Readmission"
    use_los = label == "Length of Stay"
    los = time if use_los else 0
    use_ICU = use_ICU == "ICU"

    group_col = "subject_id"
    visit_col = "stay_id" if use_ICU else "hadm_id"
    admit_col = "intime" if use_ICU else "admittime"
    disch_col = "outtime" if use_ICU else "dischtime"
    death_col = "dod"
    adm_visit_col = "hadm_id" if use_ICU else None

    pts = get_visit_pts(
        mimic4_path=root_dir + "/mimiciv/2.2/",
        group_col=group_col,
        visit_col=visit_col,
        admit_col=admit_col,
        disch_col=disch_col,
        adm_visit_col=adm_visit_col,
        use_admn=use_admn,
        disease_label=disease_label,
        use_ICU=use_ICU,
    )
    cols = [
        group_col,
        visit_col,
        admit_col,
        disch_col,
        "Age",
        "gender",
        "ethnicity",
        "insurance",
        "label",
    ]
    if use_ICU:
        cols.append(adm_visit_col)
    if use_mort:
        cols.append(death_col)

    cohort, invalid = get_case_ctrls(
        pts,
        time,
        group_col,
        visit_col,
        admit_col,
        disch_col,
        "min_valid_year",
        death_col,
        use_mort,
        use_admn,
        use_los,
    )

    if icd_code != "No Disease Filter":
        hids = disease_cohort.extract_diag_cohort(
            cohort.select("hadm_id"), icd_code, f"{root_dir}/mimiciv/2.2/"
        )
        cohort = cohort.filter(pl.col("hadm_id").is_in(hids["hadm_id"].to_list()))
        cohort_output += f"_{icd_code}"
        summary_output += f"_{icd_code}"

    cohort = cohort.rename({"race": "ethnicity"})
    cohort.select(cols).write_csv(f"./data/cohort/{cohort_output}.csv")
    print("[ COHORT SUCCESSFULLY SAVED ]")

    summary = "\n".join(
        [
            f"{label} FOR {ICU} DATA",
            f"# Admission Records: {cohort.shape[0]}",
            f"# Patients: {cohort.select(group_col).n_unique()}",  # Adjusted for Polars
            f"# Positive cases: {cohort.filter(pl.col('label') == 1).shape[0]}",
            f"# Negative cases: {cohort.filter(pl.col('label') == 0).shape[0]}",
        ]
    )

    print(summary)
    with open(f"./data/cohort/{summary_output}.txt", "w") as text_file:
        text_file.write(summary)
    print("[ SUMMARY SUCCESSFULLY SAVED ]")
    return cohort_output


def extract_imputation_data(use_ICU, root_dir, cohort_output=None, summary_output=None):
    cohort_output = cohort_output or f"cohort_{use_ICU.lower()}_imputation"
    print(f"EXTRACTING FOR: | {use_ICU.upper()} | IMPUTATION |")

    use_ICU = use_ICU == "ICU"
    group_col = "subject_id"
    admit_col = "intime" if use_ICU else "admittime"
    disch_col = "outtime" if use_ICU else "dischtime"
    mimic4_path = root_dir + "/mimiciv/2.2/"
    file_path = "icu/icustays.csv.gz" if use_ICU else "hosp/admissions.csv.gz"
    visit = pl.read_csv(
        mimic4_path + file_path,
        try_parse_dates=True,
        dtypes={admit_col: pl.Datetime, disch_col: pl.Datetime},
    )

    if not use_ICU:
        visit = visit.with_columns((pl.col(disch_col) - pl.col(admit_col)).alias("los"))

    pts = pl.read_csv(mimic4_path + "hosp/patients.csv.gz")

    pts = pts.with_columns(
        [
            (pl.col("anchor_year") - pl.col("anchor_age")).alias("yob"),
            (
                pl.col("anchor_year")
                + (2019 - pl.col("anchor_year_group").str.slice(-4).cast(pl.Int32))
            ).alias("min_valid_year"),
        ]
    )
    visit_pts = visit.join(pts, on=group_col, how="inner")

    if use_ICU:
        eth = pl.read_csv(mimic4_path + "hosp/admissions.csv.gz")
        visit_pts = visit_pts.join(eth, on="hadm_id", how="inner")
    visit_pts.write_csv(f"./data/cohort/{cohort_output}.csv")
    return cohort_output
