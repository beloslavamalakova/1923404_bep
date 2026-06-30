import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Load data
# ============================================================

train = pd.read_csv("train_features_ablation.csv", parse_dates=["timestamp"])
test = pd.read_csv("test_features_ablation.csv", parse_dates=["timestamp"])

# Combine train + test
df = pd.concat([train, test], ignore_index=True)

# ============================================================
# Extract hour
# ============================================================

df["hour"] = df["timestamp"].dt.hour

# ============================================================
# Average glucose per patient per hour
# ============================================================

avg_glucose = (
    df.groupby(["patient", "hour"])["glucose"]
    .mean()
    .reset_index()
)

patients = sorted(avg_glucose["patient"].unique())

# ============================================================
# Create subplot grid
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)

axes = axes.flatten()

# ============================================================
# Plot each patient separately
# ============================================================

for ax, patient in zip(axes, patients):

    patient_df = avg_glucose[avg_glucose["patient"] == patient]

    ax.plot(
        patient_df["hour"],
        patient_df["glucose"],
        marker="o",
        linewidth=2
    )

    ax.set_title(f"Patient {patient}", fontsize=10)
    ax.set_xticks(range(0, 24, 6))
    ax.grid(True, alpha=0.3)

# ============================================================
# Global labels
# ============================================================

fig.suptitle(
    "Average Glucose Levels Throughout the Day per Patient",
    fontsize=16
)

fig.supxlabel("Hour of Day", fontsize=12)
fig.supylabel("Average Glucose", fontsize=12)

plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

# ============================================================
# Save figure
# ============================================================

plt.savefig(
    "avg_glucose_subplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
