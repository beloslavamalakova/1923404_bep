import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

OUTPUT_DIR = "ablation_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 1. ABLATION RESULTS TABLE
# --------------------------------------------------

results = pd.DataFrame({
    "model": [
        "LSTM", "Attention-LSTM",
        "LSTM", "Attention-LSTM",
        "LSTM", "Attention-LSTM"
    ],
    "feature_set": [
        "Glucose only", "Glucose only",
        "Glucose + insulin", "Glucose + insulin",
        "Glucose + meal", "Glucose + meal"
    ],
    "MAE": [16.410454, 16.455406, 16.589853, 16.350945, 16.672958, 16.119320],
    "RMSE": [23.080012, 23.119602, 23.270222, 23.199270, 23.305486, 22.807250],
    "R2": [0.853833, 0.853331, 0.851414, 0.852319, 0.850963, 0.857268],
})

results.to_csv(os.path.join(OUTPUT_DIR, "ablation_results_clean.csv"), index=False)


# --------------------------------------------------
# 2. BAR PLOTS FOR MAE, RMSE, R2
# --------------------------------------------------

for metric in ["MAE", "RMSE", "R2"]:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results, x="feature_set", y=metric, hue="model")
    plt.title(f"Ablation Study: {metric} by Feature Set")
    plt.xlabel("Feature Set")
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_barplot.png"), dpi=300)
    plt.show()


# --------------------------------------------------
# 3. LINE PLOTS: HOW EACH MODEL CHANGES WITH FEATURES
# --------------------------------------------------

for metric in ["MAE", "RMSE", "R2"]:
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=results,
        x="feature_set",
        y=metric,
        hue="model",
        marker="o"
    )
    plt.title(f"{metric} Trend Across Feature Sets")
    plt.xlabel("Feature Set")
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_lineplot.png"), dpi=300)
    plt.show()


# --------------------------------------------------
# 4. ATTENTION IMPROVEMENT OVER LSTM
# --------------------------------------------------

pivot = results.pivot(index="feature_set", columns="model", values=["MAE", "RMSE", "R2"])

improvement = pd.DataFrame()
improvement["feature_set"] = pivot.index
improvement["MAE_improvement"] = pivot[("MAE", "LSTM")] - pivot[("MAE", "Attention-LSTM")]
improvement["RMSE_improvement"] = pivot[("RMSE", "LSTM")] - pivot[("RMSE", "Attention-LSTM")]
improvement["R2_improvement"] = pivot[("R2", "Attention-LSTM")] - pivot[("R2", "LSTM")]

improvement.to_csv(os.path.join(OUTPUT_DIR, "attention_improvement_over_lstm.csv"), index=False)

for metric in ["MAE_improvement", "RMSE_improvement", "R2_improvement"]:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=improvement, x="feature_set", y=metric)
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Attention-LSTM Improvement over LSTM: {metric}")
    plt.xlabel("Feature Set")
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}.png"), dpi=300)
    plt.show()


# --------------------------------------------------
# 5. HEATMAP OF RESULTS
# --------------------------------------------------

for metric in ["MAE", "RMSE", "R2"]:
    heatmap_data = results.pivot(index="model", columns="feature_set", values=metric)

    plt.figure(figsize=(8, 4))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"Heatmap of {metric}")
    plt.xlabel("Feature Set")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_heatmap.png"), dpi=300)
    plt.show()


# --------------------------------------------------
# 6. LOAD PREDICTION FILES IF THEY EXIST
# --------------------------------------------------

prediction_files = glob.glob("*predictions*.csv") + glob.glob("outputs/*predictions*.csv")

print("\nFound prediction files:")
for f in prediction_files:
    print(f)

for file in prediction_files:
    try:
        df = pd.read_csv(file)

        true_cols = [c for c in df.columns if "true" in c.lower()]
        pred_cols = [c for c in df.columns if "pred" in c.lower()]

        if not true_cols or not pred_cols:
            continue

        true_col = true_cols[0]
        pred_col = pred_cols[0]

        name = os.path.basename(file).replace(".csv", "")

        # true vs predicted scatter
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=df.sample(min(len(df), 5000), random_state=42), x=true_col, y=pred_col, alpha=0.3)
        min_val = min(df[true_col].min(), df[pred_col].min())
        max_val = max(df[true_col].max(), df[pred_col].max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
        plt.title(f"True vs Predicted Glucose: {name}")
        plt.xlabel("True glucose")
        plt.ylabel("Predicted glucose")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_true_vs_pred.png"), dpi=300)
        plt.show()

        # residual plot
        df["residual"] = df[true_col] - df[pred_col]

        plt.figure(figsize=(8, 5))
        sns.histplot(df["residual"], bins=80, kde=True)
        plt.title(f"Residual Distribution: {name}")
        plt.xlabel("True - Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_residual_distribution.png"), dpi=300)
        plt.show()

        # time-series example for one patient
        if "patient" in df.columns and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            example_patient = df["patient"].iloc[0]
            patient_df = df[df["patient"] == example_patient].sort_values("timestamp").head(300)

            plt.figure(figsize=(12, 5))
            plt.plot(patient_df["timestamp"], patient_df[true_col], label="True glucose")
            plt.plot(patient_df["timestamp"], patient_df[pred_col], label="Predicted glucose")
            plt.title(f"Prediction Over Time: {name}, Patient {example_patient}")
            plt.xlabel("Timestamp")
            plt.ylabel("Glucose")
            plt.legend()
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_time_series_patient_{example_patient}.png"), dpi=300)
            plt.show()

    except Exception as e:
        print(f"Could not process {file}: {e}")


# --------------------------------------------------
# 7. ATTENTION WEIGHTS IF AVAILABLE
# --------------------------------------------------

attention_files = glob.glob("*attention_weights*.csv") + glob.glob("outputs/*attention_weights*.csv")

print("\nFound attention weight files:")
for f in attention_files:
    print(f)

for file in attention_files:
    try:
        attn = pd.read_csv(file)
        name = os.path.basename(file).replace(".csv", "")

        if "lag" in attn.columns and "average_attention_weight" in attn.columns:
            plt.figure(figsize=(9, 5))
            sns.barplot(data=attn, x="lag", y="average_attention_weight")
            plt.title(f"Average Attention Weights: {name}")
            plt.xlabel("Glucose lag")
            plt.ylabel("Average attention weight")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_attention_weights.png"), dpi=300)
            plt.show()

        else:
            numeric_cols = attn.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                avg_weights = attn[numeric_cols].mean()

                plt.figure(figsize=(9, 5))
                avg_weights.plot(kind="bar")
                plt.title(f"Average Attention Weights: {name}")
                plt.xlabel("Timestep")
                plt.ylabel("Average attention weight")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_attention_weights.png"), dpi=300)
                plt.show()

    except Exception as e:
        print(f"Could not process {file}: {e}")


# --------------------------------------------------
# 8. FEATURE DISTRIBUTIONS FROM ABLATION FEATURES
# --------------------------------------------------

if os.path.exists("train_features_ablation.csv"):
    train = pd.read_csv("train_features_ablation.csv")

    feature_cols = [
        "glucose", "basal", "bolus", "carbs",
        "bolus_30min", "carbs_30min",
        "time_since_last_bolus_min", "time_since_last_meal_min",
        "target"
    ]

    for col in feature_cols:
        if col in train.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(train[col], bins=80, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"distribution_{col}.png"), dpi=300)
            plt.show()

    # glucose by hour
    if "timestamp" in train.columns:
        train["timestamp"] = pd.to_datetime(train["timestamp"])
        train["hour"] = train["timestamp"].dt.hour

        hourly = train.groupby("hour")["glucose"].mean().reset_index()

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=hourly, x="hour", y="glucose", marker="o")
        plt.title("Average Glucose by Hour of Day")
        plt.xlabel("Hour")
        plt.ylabel("Mean glucose")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "average_glucose_by_hour.png"), dpi=300)
        plt.show()

    # glucose distribution per patient
    if "patient" in train.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=train, x="patient", y="glucose")
        plt.title("Glucose Distribution per Patient")
        plt.xlabel("Patient")
        plt.ylabel("Glucose")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "glucose_distribution_per_patient.png"), dpi=300)
        plt.show()


print(f"\nAll visualizations saved in: {OUTPUT_DIR}/")
