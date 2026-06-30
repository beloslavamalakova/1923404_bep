import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train_features.csv", parse_dates=["timestamp"])
test = pd.read_csv("test_features.csv", parse_dates=["timestamp"])

df = pd.concat([train, test], ignore_index=True)


df["hour"] = df["timestamp"].dt.hour

avg_glucose = (
    df.groupby(["patient", "hour"])["glucose"]
    .mean()
    .reset_index()
)


plt.figure(figsize=(12, 7))

for patient in sorted(avg_glucose["patient"].unique()):
    patient_df = avg_glucose[avg_glucose["patient"] == patient]

    plt.plot(
        patient_df["hour"],
        patient_df["glucose"],
        marker="o",
        linewidth=2,
        label=f"Patient {patient}"
    )

plt.xlabel("Hour of Day")
plt.ylabel("Average Glucose")
plt.title("Average Glucose Levels Throughout the Day per Patient")

plt.xticks(range(24))
plt.grid(True, alpha=0.3)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    "avg_glucose_per_patient_per_day.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
