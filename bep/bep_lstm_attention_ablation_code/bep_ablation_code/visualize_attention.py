import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "attention_visualizations"
ATTN_DIR = "ablation_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 1. LOAD ATTENTION WEIGHT FILES
# --------------------------------------------------

attention_files = sorted(glob.glob(os.path.join(ATTN_DIR, "attention_weights_*.csv")))

if not attention_files:
    raise FileNotFoundError("No attention weight files found in ablation_outputs/")

all_attention = []

for file in attention_files:
    df = pd.read_csv(file)

    # Fix for older attention files that only have:
    # lag, average_attention_weight
    if "minutes_before_prediction" not in df.columns:
        if "lag" not in df.columns:
            raise ValueError(f"{file} has no 'lag' column and no 'minutes_before_prediction' column.")

        def lag_to_minutes(lag_name):
            lag_num = int(str(lag_name).split("_")[-1])
            return lag_num * 5

        df["minutes_before_prediction"] = df["lag"].apply(lag_to_minutes)

    feature_set = (
        os.path.basename(file)
        .replace("attention_weights_", "")
        .replace(".csv", "")
    )

    df["feature_set"] = feature_set
    all_attention.append(df)

attention_df = pd.concat(all_attention, ignore_index=True)

attention_df.to_csv(
    os.path.join(OUTPUT_DIR, "all_attention_weights.csv"),
    index=False
)


# --------------------------------------------------
# 2. ATTENTION WEIGHTS PER FEATURE SET
# --------------------------------------------------

for feature_set in attention_df["feature_set"].unique():
    df = attention_df[attention_df["feature_set"] == feature_set].copy()
    df = df.sort_values("minutes_before_prediction", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df["minutes_before_prediction"], df["average_attention_weight"])
    plt.gca().invert_xaxis()
    plt.title(f"Average Attention Weights: {feature_set}")
    plt.xlabel("Minutes before prediction")
    plt.ylabel("Average attention weight")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"attention_weights_{feature_set}.png"),
        dpi=300
    )
    plt.show()


# --------------------------------------------------
# 3. COMPARE ATTENTION WEIGHTS ACROSS FEATURE SETS
# --------------------------------------------------

plt.figure(figsize=(9, 5))

for feature_set in attention_df["feature_set"].unique():
    df = attention_df[attention_df["feature_set"] == feature_set].copy()
    df = df.sort_values("minutes_before_prediction", ascending=False)

    plt.plot(
        df["minutes_before_prediction"],
        df["average_attention_weight"],
        marker="o",
        label=feature_set
    )

plt.gca().invert_xaxis()
plt.title("Average Attention Weights Across Feature Sets")
plt.xlabel("Minutes before prediction")
plt.ylabel("Average attention weight")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "attention_weights_comparison.png"),
    dpi=300
)
plt.show()


# --------------------------------------------------
# 4. ATTENTION HEATMAP
# --------------------------------------------------

heatmap_df = attention_df.pivot(
    index="feature_set",
    columns="minutes_before_prediction",
    values="average_attention_weight"
)

heatmap_df = heatmap_df[sorted(heatmap_df.columns, reverse=True)]

plt.figure(figsize=(10, 4))
plt.imshow(heatmap_df.values, aspect="auto")
plt.colorbar(label="Average attention weight")
plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns)
plt.xlabel("Minutes before prediction")
plt.ylabel("Feature set")
plt.title("Attention Weight Heatmap")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "attention_weight_heatmap.png"),
    dpi=300
)
plt.show()


# --------------------------------------------------
# 5. FEATURE DISTRIBUTIONS
# --------------------------------------------------

if os.path.exists("train_features_ablation.csv"):
    train = pd.read_csv("train_features_ablation.csv")

    feature_cols = [
        "glucose",
        "basal",
        "bolus",
        "bolus_30min",
        "carbs",
        "carbs_30min",
        "time_since_last_bolus_min",
        "time_since_last_meal_min",
        "target"
    ]

    for col in feature_cols:
        if col in train.columns:
            plt.figure(figsize=(8, 5))
            plt.hist(train[col].dropna(), bins=80)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, f"distribution_{col}.png"),
                dpi=300
            )
            plt.show()


# --------------------------------------------------
# 6. FEATURE VS TARGET RELATIONSHIPS
# --------------------------------------------------

if os.path.exists("train_features_ablation.csv"):
    train = pd.read_csv("train_features_ablation.csv")

    scatter_features = [
        "glucose",
        "basal",
        "bolus",
        "bolus_30min",
        "carbs",
        "carbs_30min",
        "time_since_last_bolus_min",
        "time_since_last_meal_min"
    ]

    sample = train.sample(min(len(train), 5000), random_state=42)

    for col in scatter_features:
        if col in sample.columns:
            plt.figure(figsize=(7, 5))
            plt.scatter(sample[col], sample["target"], alpha=0.25, s=8)
            plt.title(f"{col} vs 30-minute Future Glucose")
            plt.xlabel(col)
            plt.ylabel("Target glucose")
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, f"{col}_vs_target.png"),
                dpi=300
            )
            plt.show()


# --------------------------------------------------
# 7. PREDICTION ERRORS FOR ATTENTION-LSTM
# --------------------------------------------------

prediction_files = sorted(
    glob.glob(os.path.join(ATTN_DIR, "predictions_attention_lstm_*.csv"))
)

for file in prediction_files:
    df = pd.read_csv(file)

    feature_set = (
        os.path.basename(file)
        .replace("predictions_attention_lstm_", "")
        .replace(".csv", "")
    )

    true_col = "true_glucose"
    pred_col = "predicted_glucose"

    if true_col not in df.columns or pred_col not in df.columns:
        print(f"Skipping {file}, could not find true/predicted glucose columns.")
        continue

    df["error"] = df[true_col] - df[pred_col]
    df["abs_error"] = df["error"].abs()

    plt.figure(figsize=(8, 5))
    plt.hist(df["error"], bins=80)
    plt.title(f"Prediction Error Distribution: Attention-LSTM {feature_set}")
    plt.xlabel("True glucose - predicted glucose")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"error_distribution_attention_{feature_set}.png"),
        dpi=300
    )
    plt.show()

    plt.figure(figsize=(6, 6))
    sample = df.sample(min(len(df), 5000), random_state=42)
    plt.scatter(sample[true_col], sample[pred_col], alpha=0.25, s=8)

    min_val = min(sample[true_col].min(), sample[pred_col].min())
    max_val = max(sample[true_col].max(), sample[pred_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.title(f"True vs Predicted: Attention-LSTM {feature_set}")
    plt.xlabel("True glucose")
    plt.ylabel("Predicted glucose")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"true_vs_pred_attention_{feature_set}.png"),
        dpi=300
    )
    plt.show()


print(f"\nSaved all plots in: {OUTPUT_DIR}/")
