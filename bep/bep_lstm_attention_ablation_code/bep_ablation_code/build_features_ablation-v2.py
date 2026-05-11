import os
import glob
import argparse
import numpy as np
import pandas as pd
from lxml import etree
from tqdm import tqdm


YEARS = ["2018", "2020"]


def parse_patient_xml(filepath: str):
    tree = etree.parse(filepath)
    root = tree.getroot()
    patient_id = root.get("id")

    data = {
        "glucose": [],
        "basal": [],
        "bolus": [],
        "meal": [],
    }

    glucose_block = root.find("glucose_level")
    if glucose_block is not None:
        for e in glucose_block.findall("event"):
            ts = e.get("ts")
            value = e.get("value")
            if ts is not None and value is not None:
                data["glucose"].append((patient_id, ts, float(value)))

    basal_block = root.find("basal")
    if basal_block is not None:
        for e in basal_block.findall("event"):
            ts = e.get("ts")
            value = e.get("value")
            if ts is not None and value is not None:
                data["basal"].append((patient_id, ts, float(value)))

    bolus_block = root.find("bolus")
    if bolus_block is not None:
        for e in bolus_block.findall("event"):
            ts = e.get("ts_begin") or e.get("ts")
            dose = e.get("dose")
            if ts is not None and dose is not None:
                data["bolus"].append((patient_id, ts, float(dose)))

    meal_block = root.find("meal")
    if meal_block is not None:
        for e in meal_block.findall("event"):
            ts = e.get("ts")
            carbs = e.get("carbs")
            if ts is not None and carbs is not None:
                data["meal"].append((patient_id, ts, float(carbs)))

    return data


def _to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")
    return df


def load_split(base_path: str, split: str):
    all_glucose, all_basal, all_bolus, all_meals = [], [], [], []

    for year in YEARS:
        folder = os.path.join(base_path, year, split)
        files = sorted(glob.glob(os.path.join(folder, "*.xml")))
        if not files:
            print(f"Warning: no XML files found in {folder}")

        for file in tqdm(files, desc=f"{year}-{split}"):
            d = parse_patient_xml(file)
            all_glucose.extend(d["glucose"])
            all_basal.extend(d["basal"])
            all_bolus.extend(d["bolus"])
            all_meals.extend(d["meal"])

    glucose_df = pd.DataFrame(all_glucose, columns=["patient", "timestamp", "glucose"])
    basal_df = pd.DataFrame(all_basal, columns=["patient", "timestamp", "basal"])
    bolus_df = pd.DataFrame(all_bolus, columns=["patient", "timestamp", "bolus"])
    meal_df = pd.DataFrame(all_meals, columns=["patient", "timestamp", "carbs"])

    for df in [glucose_df, basal_df, bolus_df, meal_df]:
        _to_datetime(df)

    return glucose_df, basal_df, bolus_df, meal_df


def build_regular_glucose_table(glucose_df, basal_df, bolus_df, meal_df):
    """Merge event streams onto the 5-minute CGM grid patient by patient.

    FIX: bolus and meal events are now aggregated into 5-min bins before
    merging, so every event is captured regardless of whether it lands
    exactly on a CGM timestamp.  The old merge_asof with a 5-min tolerance
    silently dropped most events, leaving bolus/carbs nearly always zero.
    """
    patient_tables = []

    for patient in sorted(glucose_df["patient"].unique()):
        g = glucose_df[glucose_df["patient"] == patient].sort_values("timestamp").copy()
        b = basal_df[basal_df["patient"] == patient].sort_values("timestamp").copy()
        bo = bolus_df[bolus_df["patient"] == patient].sort_values("timestamp").copy()
        m = meal_df[meal_df["patient"] == patient].sort_values("timestamp").copy()

        merged = g[["timestamp", "glucose"]].copy()

        # Basal: use backward-fill (step function) — unchanged, correct behaviour.
        if not b.empty:
            merged = pd.merge_asof(merged, b[["timestamp", "basal"]], on="timestamp", direction="backward")
        else:
            merged["basal"] = np.nan

        # FIX: resample bolus events into 5-min bins (summing within each bin)
        # then merge on the rounded CGM timestamps so nothing is dropped.
        if not bo.empty:
            # Round bolus timestamps to the nearest 5-min boundary.
            bo_binned = bo.copy()
            bo_binned["timestamp"] = bo_binned["timestamp"].dt.round("5min")
            bo_binned = bo_binned.groupby("timestamp", as_index=False)["bolus"].sum()
            merged = pd.merge_asof(
                merged,
                bo_binned[["timestamp", "bolus"]],
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("5min"),
            )
        else:
            merged["bolus"] = 0.0

        # FIX: same binning fix for meal / carbs.
        if not m.empty:
            m_binned = m.copy()
            m_binned["timestamp"] = m_binned["timestamp"].dt.round("5min")
            m_binned = m_binned.groupby("timestamp", as_index=False)["carbs"].sum()
            merged = pd.merge_asof(
                merged,
                m_binned[["timestamp", "carbs"]],
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("5min"),
            )
        else:
            merged["carbs"] = 0.0

        merged["patient"] = patient
        merged["basal"] = merged["basal"].ffill().fillna(0.0)
        merged["bolus"] = merged["bolus"].fillna(0.0)
        merged["carbs"] = merged["carbs"].fillna(0.0)

        patient_tables.append(merged[["patient", "timestamp", "glucose", "basal", "bolus", "carbs"]])

    full_df = pd.concat(patient_tables, ignore_index=True)
    return full_df.sort_values(["patient", "timestamp"]).reset_index(drop=True)


def add_features(df: pd.DataFrame, horizon_steps: int = 6, history_steps: int = 12):
    df = df.sort_values(["patient", "timestamp"]).copy()

    # Glucose history: lag_12 is oldest, lag_1 is most recent.
    for lag in range(1, history_steps + 1):
        df[f"glucose_lag_{lag}"] = df.groupby("patient")["glucose"].shift(lag)

    # 30-minute default target: 6 steps ahead with 5-minute sampling.
    df["target"] = df.groupby("patient")["glucose"].shift(-horizon_steps)

    # Rolling intervention summaries.
    # These are now meaningful because bolus/carbs are no longer mostly zero.
    df["bolus_30min"] = (
        df.groupby("patient")["bolus"]
        .rolling(window=6, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["carbs_30min"] = (
        df.groupby("patient")["carbs"]
        .rolling(window=6, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # Timing features: minutes since most recent bolus/meal.
    df["last_bolus_time"] = df["timestamp"].where(df["bolus"] > 0)
    df["last_bolus_time"] = df.groupby("patient")["last_bolus_time"].ffill()
    df["time_since_last_bolus_min"] = (
        (df["timestamp"] - df["last_bolus_time"]).dt.total_seconds() / 60.0
    )

    df["last_meal_time"] = df["timestamp"].where(df["carbs"] > 0)
    df["last_meal_time"] = df.groupby("patient")["last_meal_time"].ffill()
    df["time_since_last_meal_min"] = (
        (df["timestamp"] - df["last_meal_time"]).dt.total_seconds() / 60.0
    )

    # FIX: cap at 480 min (8 hours) instead of 9999.
    # 9999 was an extreme outlier that distorted StandardScaler; physiologically
    # nothing meaningful changes after ~8 hours without insulin or food.
    df["time_since_last_bolus_min"] = df["time_since_last_bolus_min"].fillna(480.0).clip(0, 480)
    df["time_since_last_meal_min"] = df["time_since_last_meal_min"].fillna(480.0).clip(0, 480)

    df = df.drop(columns=["last_bolus_time", "last_meal_time"])
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="data", help="Path to OhioT1DM data folder")
    parser.add_argument("--horizon-steps", type=int, default=6, help="Prediction horizon in 5-min steps; 6 = 30 min")
    parser.add_argument("--history-steps", type=int, default=12, help="Glucose history length in 5-min steps; 12 = 60 min")
    args = parser.parse_args()

    for split in ["train", "test"]:
        print(f"\nBuilding {split} split...")
        g, b, bo, m = load_split(args.base_path, split)
        model_df = build_regular_glucose_table(g, b, bo, m)
        feat_df = add_features(model_df, horizon_steps=args.horizon_steps, history_steps=args.history_steps)
        out_path = f"{split}_features_ablation.csv"
        feat_df.to_csv(out_path, index=False)
        print(f"Saved {out_path}: {feat_df.shape}")

        # Sanity-check: report how sparse the intervention columns actually are.
        bolus_nonzero = (feat_df["bolus"] > 0).mean() * 100
        carbs_nonzero = (feat_df["carbs"] > 0).mean() * 100
        print(f"  bolus > 0: {bolus_nonzero:.1f}% of rows  |  carbs > 0: {carbs_nonzero:.1f}% of rows")


if __name__ == "__main__":
    main()
