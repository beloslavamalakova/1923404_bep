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

    patient_id = root.get("id")  # now safe, since root = <patient>

    data = {
        "glucose": [],
        "basal": [],
        "bolus": [],
        "meal": []
    }

    # GLUCOSE
    glucose_block = root.find("glucose_level")
    if glucose_block is not None:
        for event in glucose_block.findall("event"):
            ts = event.get("ts")
            value = event.get("value")
            if ts and value:
                data["glucose"].append((patient_id, ts, float(value)))

    # BASAL
    basal_block = root.find("basal")
    if basal_block is not None:
        for event in basal_block.findall("event"):
            ts = event.get("ts")
            value = event.get("value")
            if ts and value:
                data["basal"].append((patient_id, ts, float(value)))

    # BOLUS
    bolus_block = root.find("bolus")
    if bolus_block is not None:
        for event in bolus_block.findall("event"):
            ts = event.get("ts_begin")
            dose = event.get("dose")
            if ts and dose:
                data["bolus"].append((patient_id, ts, float(dose)))

    # MEAL
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


# CREATE DATAFRAMES

glucose_df = pd.DataFrame(all_glucose, columns=["patient", "timestamp", "glucose"])
basal_df = pd.DataFrame(all_basal, columns=["patient", "timestamp", "basal"])
bolus_df = pd.DataFrame(all_bolus, columns=["patient", "timestamp", "bolus"])
meal_df = pd.DataFrame(all_meals, columns=["patient", "timestamp", "carbs"])

# Convert timestamps
for df in [glucose_df, basal_df, bolus_df, meal_df]:
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")


print("\n===== DATA OVERVIEW =====")
print("Glucose:", glucose_df.shape)
print("Basal:", basal_df.shape)
print("Bolus:", bolus_df.shape)
print("Meal:", meal_df.shape)

print("\nGlucose summary statistics:")
print(glucose_df["glucose"].describe())


# --------------------------------------------------
# GLUCOSE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(glucose_df["glucose"], bins=100, kde=True)
plt.title("Glucose Distribution")
plt.xlabel("Glucose (mg/dL)")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# HYPO / HYPER RATES
# --------------------------------------------------

hypo_rate = (glucose_df["glucose"] < 70).mean()
hyper_rate = (glucose_df["glucose"] > 180).mean()

print(f"\nHypoglycemia rate (<70 mg/dL): {hypo_rate:.3f}")
print(f"Hyperglycemia rate (>180 mg/dL): {hyper_rate:.3f}")


# --------------------------------------------------
# HOURLY GLUCOSE PATTERN
# --------------------------------------------------

glucose_df["hour"] = glucose_df["timestamp"].dt.hour
hourly_mean = glucose_df.groupby("hour")["glucose"].mean()

plt.figure(figsize=(8,5))
hourly_mean.plot()
plt.title("Average Glucose by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Glucose")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# PER PATIENT STATS
# --------------------------------------------------

patient_stats = glucose_df.groupby("patient")["glucose"].agg(["mean", "std"])

print("\nPer-patient glucose statistics:")
print(patient_stats)

plt.figure(figsize=(10,5))
sns.boxplot(data=glucose_df, x="patient", y="glucose")
plt.xticks(rotation=45)
plt.title("Glucose Distribution per Patient")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# MEAL CARBS DISTRIBUTION
# --------------------------------------------------

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
