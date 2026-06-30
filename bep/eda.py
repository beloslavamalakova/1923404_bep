import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lxml import etree
from tqdm import tqdm

BASE_PATH = "data"
YEARS = ["2018", "2020"]
SPLITS = ["train", "test"]

sns.set(style="whitegrid")


def parse_patient_xml(filepath):
    tree = etree.parse(filepath)
    root = tree.getroot()

    patient_id = root.get("id")  

    data = {
        "glucose": [],
        "basal": [],
        "bolus": [],
        "meal": []
    }

    
    glucose_block = root.find("glucose_level")
    if glucose_block is not None:
        for event in glucose_block.findall("event"):
            ts = event.get("ts")
            value = event.get("value")
            if ts and value:
                data["glucose"].append((patient_id, ts, float(value)))

    
    basal_block = root.find("basal")
    if basal_block is not None:
        for event in basal_block.findall("event"):
            ts = event.get("ts")
            value = event.get("value")
            if ts and value:
                data["basal"].append((patient_id, ts, float(value)))

    
    bolus_block = root.find("bolus")
    if bolus_block is not None:
        for event in bolus_block.findall("event"):
            ts = event.get("ts_begin")
            dose = event.get("dose")
            if ts and dose:
                data["bolus"].append((patient_id, ts, float(dose)))

    
    meal_block = root.find("meal")
    if meal_block is not None:
        for event in meal_block.findall("event"):
            ts = event.get("ts")
            carbs = event.get("carbs")
            if ts and carbs:
                data["meal"].append((patient_id, ts, float(carbs)))

    return data


all_glucose = []
all_basal = []
all_bolus = []
all_meals = []

for year in YEARS:
    for split in SPLITS:
        folder = os.path.join(BASE_PATH, year, split)
        files = glob.glob(os.path.join(folder, "*.xml"))

        for file in tqdm(files, desc=f"{year}-{split}"):
            patient_data = parse_patient_xml(file)

            all_glucose.extend(patient_data["glucose"])
            all_basal.extend(patient_data["basal"])
            all_bolus.extend(patient_data["bolus"])
            all_meals.extend(patient_data["meal"])




glucose_df = pd.DataFrame(all_glucose, columns=["patient", "timestamp", "glucose"])
basal_df = pd.DataFrame(all_basal, columns=["patient", "timestamp", "basal"])
bolus_df = pd.DataFrame(all_bolus, columns=["patient", "timestamp", "bolus"])
meal_df = pd.DataFrame(all_meals, columns=["patient", "timestamp", "carbs"])


for df in [glucose_df, basal_df, bolus_df, meal_df]:
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")


print("\n===== DATA OVERVIEW =====")
print("Glucose:", glucose_df.shape)
print("Basal:", basal_df.shape)
print("Bolus:", bolus_df.shape)
print("Meal:", meal_df.shape)

print("\nGlucose summary statistics:")
print(glucose_df["glucose"].describe())






plt.figure(figsize=(8,5))
sns.histplot(glucose_df["glucose"], bins=100, kde=True)
plt.title("Glucose Distribution")
plt.xlabel("Glucose (mg/dL)")
plt.tight_layout()
plt.show()






hypo_rate = (glucose_df["glucose"] < 70).mean()
hyper_rate = (glucose_df["glucose"] > 180).mean()

print(f"\nHypoglycemia rate (<70 mg/dL): {hypo_rate:.3f}")
print(f"Hyperglycemia rate (>180 mg/dL): {hyper_rate:.3f}")






glucose_df["hour"] = glucose_df["timestamp"].dt.hour
hourly_mean = glucose_df.groupby("hour")["glucose"].mean()

plt.figure(figsize=(8,5))
hourly_mean.plot()
plt.title("Average Glucose by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Glucose")
plt.tight_layout()
plt.show()






patient_stats = glucose_df.groupby("patient")["glucose"].agg(["mean", "std"])

print("\nPer-patient glucose statistics:")
print(patient_stats)

plt.figure(figsize=(10,5))
sns.boxplot(data=glucose_df, x="patient", y="glucose")
plt.xticks(rotation=45)
plt.title("Glucose Distribution per Patient")
plt.tight_layout()
plt.show()






if not meal_df.empty:
    print("\nMeal carbohydrate summary:")
    print(meal_df["carbs"].describe())

    plt.figure(figsize=(8,5))
    sns.histplot(meal_df["carbs"], bins=40, kde=True)
    plt.title("Meal Carbohydrate Distribution")
    plt.xlabel("Carbs (grams)")
    plt.tight_layout()
    plt.show()


print("\nEDA completed successfully.")





print("\n===== CGM MISSINGNESS / SENSOR GAP ANALYSIS =====")


glucose_gap_df = glucose_df.sort_values(["patient", "timestamp"]).copy()


glucose_gap_df["time_diff_minutes"] = (
    glucose_gap_df.groupby("patient")["timestamp"]
    .diff()
    .dt.total_seconds() / 60
)


expected_interval = 5


glucose_gaps = glucose_gap_df[
    glucose_gap_df["time_diff_minutes"] > expected_interval
].copy()


glucose_gaps["estimated_missing_cgm_points"] = (
    glucose_gaps["time_diff_minutes"] / expected_interval - 1
).round().astype(int)

total_observed_cgm = len(glucose_df)
total_missing_cgm = glucose_gaps["estimated_missing_cgm_points"].sum()
total_expected_cgm = total_observed_cgm + total_missing_cgm

missing_cgm_percentage = (
    100 * total_missing_cgm / total_expected_cgm
    if total_expected_cgm > 0 else 0
)

print(f"Observed CGM measurements: {total_observed_cgm}")
print(f"Estimated missing CGM measurements: {total_missing_cgm}")
print(f"Estimated total expected CGM measurements: {total_expected_cgm}")
print(f"Estimated missing CGM percentage: {missing_cgm_percentage:.4f}%")


gap_summary = (
    glucose_gaps.groupby("patient")
    .agg(
        number_of_gaps=("time_diff_minutes", "count"),
        mean_gap_minutes=("time_diff_minutes", "mean"),
        max_gap_minutes=("time_diff_minutes", "max"),
        estimated_missing_cgm_points=("estimated_missing_cgm_points", "sum")
    )
    .reset_index()
)


all_patients = pd.DataFrame({"patient": sorted(glucose_df["patient"].unique())})
gap_summary = all_patients.merge(gap_summary, on="patient", how="left").fillna(0)


observed_per_patient = (
    glucose_df.groupby("patient")
    .size()
    .reset_index(name="observed_cgm_measurements")
)

gap_summary = gap_summary.merge(observed_per_patient, on="patient", how="left")
gap_summary["expected_cgm_measurements"] = (
    gap_summary["observed_cgm_measurements"]
    + gap_summary["estimated_missing_cgm_points"]
)

gap_summary["missing_cgm_percentage"] = (
    100
    * gap_summary["estimated_missing_cgm_points"]
    / gap_summary["expected_cgm_measurements"]
)

print("\nCGM gap summary per patient:")
print(gap_summary)


gap_summary.to_csv("cgm_gap_summary_per_patient.csv", index=False)
glucose_gaps.to_csv("cgm_detected_gaps.csv", index=False)

print("\nSaved:")
print("- cgm_gap_summary_per_patient.csv")
print("- cgm_detected_gaps.csv")


print("\nLargest detected CGM gaps:")
print(
    glucose_gaps[
        ["patient", "timestamp", "time_diff_minutes", "estimated_missing_cgm_points"]
    ]
    .sort_values("time_diff_minutes", ascending=False)
    .head(20)
)
