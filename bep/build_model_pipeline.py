import os
import glob
import pandas as pd
from lxml import etree
from tqdm import tqdm

BASE_PATH = "data"
YEARS = ["2018", "2020"]


def parse_patient_xml(filepath):
    tree = etree.parse(filepath)
    root = tree.getroot()

    patient_id = root.get("id")

    data = {
        "glucose": [],
        "basal": [],
        "bolus": []
    }

    # GLUCOSE
    g_block = root.find("glucose_level")
    if g_block is not None:
        for e in g_block.findall("event"):
            data["glucose"].append(
                (patient_id, e.get("ts"), float(e.get("value")))
            )

    # BASAL
    b_block = root.find("basal")
    if b_block is not None:
        for e in b_block.findall("event"):
            data["basal"].append(
                (patient_id, e.get("ts"), float(e.get("value")))
            )

    # BOLUS
    bo_block = root.find("bolus")
    if bo_block is not None:
        for e in bo_block.findall("event"):
            ts = e.get("ts_begin")
            dose = e.get("dose")
            if ts and dose:
                data["bolus"].append(
                    (patient_id, ts, float(dose))
                )

    return data


def load_split(split):

    all_glucose = []
    all_basal = []
    all_bolus = []

    for year in YEARS:
        folder = os.path.join(BASE_PATH, year, split)
        files = glob.glob(os.path.join(folder, "*.xml"))

        for file in tqdm(files, desc=f"{year}-{split}"):
            d = parse_patient_xml(file)
            all_glucose.extend(d["glucose"])
            all_basal.extend(d["basal"])
            all_bolus.extend(d["bolus"])

    glucose_df = pd.DataFrame(all_glucose, columns=["patient", "timestamp", "glucose"])
    basal_df = pd.DataFrame(all_basal, columns=["patient", "timestamp", "basal"])
    bolus_df = pd.DataFrame(all_bolus, columns=["patient", "timestamp", "bolus"])

    for df in [glucose_df, basal_df, bolus_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")

    return glucose_df, basal_df, bolus_df


def build_model_df(glucose_df, basal_df, bolus_df):

    df_list = []

    for patient in glucose_df["patient"].unique():

        g = glucose_df[glucose_df["patient"] == patient].sort_values("timestamp")

        b = basal_df[basal_df["patient"] == patient].sort_values("timestamp")
        bo = bolus_df[bolus_df["patient"] == patient].sort_values("timestamp")

        merged = pd.merge_asof(
            g,
            b,
            on="timestamp",
            direction="backward"
        )

        merged = pd.merge_asof(
            merged,
            bo,
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta("5min")
        )

        merged["bolus"] = merged["bolus"].fillna(0)
        merged["basal"] = merged["basal"].fillna(method="ffill")

        merged["patient"] = patient
        df_list.append(merged)

    full_df = pd.concat(df_list)
    full_df = full_df.sort_values(["patient", "timestamp"])

    return full_df

if __name__ == "__main__":

    print("Loading TRAIN data...")
    g_train, b_train, bo_train = load_split("train")
    train_df = build_model_df(g_train, b_train, bo_train)

    print("Loading TEST data...")
    g_test, b_test, bo_test = load_split("test")
    test_df = build_model_df(g_test, b_test, bo_test)

    train_df.to_csv("model_train.csv", index=False)
    test_df.to_csv("model_test.csv", index=False)

    print("\nSaved:")
    print("model_train.csv")
    print("model_test.csv")
