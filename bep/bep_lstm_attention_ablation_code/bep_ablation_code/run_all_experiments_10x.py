import os
import subprocess
import pandas as pd
import numpy as np

from scipy.stats import ttest_rel, wilcoxon
from tqdm import tqdm


N_RUNS = 10
SCRIPT = "run_lstm_attention_ablation-v2.py"
FINAL_OUTPUT_DIR = "final_lstm_attention_10x_results"

os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

all_runs = []


def run_command(command):
    print(f"\nExecuting:\n{command}\n")
    subprocess.run(command, shell=True, check=True)
    print("Finished.")


# ============================================================
# RUN LSTM + ATTENTION-LSTM 10 TIMES
# ============================================================

for seed in tqdm(range(N_RUNS), desc="Overall seeds"):

    print("\n" + "=" * 70)
    print(f"RUNNING SEED {seed}")
    print("=" * 70)

    output_dir = f"ablation_outputs_seed_{seed}"

    cmd = (
        f"python {SCRIPT} "
        f"--seed {seed} "
        f"--output-dir {output_dir}"
    )

    run_command(cmd)

    results_path = os.path.join(output_dir, "ablation_results.csv")
    results = pd.read_csv(results_path)
    results["seed"] = seed

    all_runs.append(results)


# ============================================================
# COMBINE RAW RESULTS
# ============================================================

all_results = pd.concat(all_runs, ignore_index=True)

raw_path = os.path.join(FINAL_OUTPUT_DIR, "all_runs_raw.csv")
all_results.to_csv(raw_path, index=False)


# ============================================================
# MEAN ± STD TABLE
# ============================================================

summary = (
    all_results
    .groupby(["model", "feature_set"])
    .agg({
        "MAE": ["mean", "std"],
        "RMSE": ["mean", "std"],
        "R2": ["mean", "std"],
    })
)

summary.columns = [
    "MAE_mean", "MAE_std",
    "RMSE_mean", "RMSE_std",
    "R2_mean", "R2_std",
]

summary = summary.reset_index()

summary["MAE"] = summary.apply(
    lambda x: f"{x['MAE_mean']:.3f} ± {x['MAE_std']:.3f}",
    axis=1
)

summary["RMSE"] = summary.apply(
    lambda x: f"{x['RMSE_mean']:.3f} ± {x['RMSE_std']:.3f}",
    axis=1
)

summary["R2"] = summary.apply(
    lambda x: f"{x['R2_mean']:.3f} ± {x['R2_std']:.3f}",
    axis=1
)

final_table = summary[
    ["model", "feature_set", "MAE", "RMSE", "R2"]
]

print("\n" + "=" * 70)
print("FINAL RESULTS: MEAN ± STD")
print("=" * 70)
print(final_table.to_string(index=False))

final_table.to_csv(
    os.path.join(FINAL_OUTPUT_DIR, "final_results_mean_std.csv"),
    index=False
)

summary.to_csv(
    os.path.join(FINAL_OUTPUT_DIR, "final_results_numeric.csv"),
    index=False
)


# ============================================================
# STATISTICAL TESTING: LSTM VS ATTENTION-LSTM
# ============================================================

significance_rows = []

print("\n" + "=" * 70)
print("STATISTICAL TESTS: LSTM VS ATTENTION-LSTM")
print("=" * 70)

for feature_set in sorted(all_results["feature_set"].unique()):

    lstm = (
        all_results[
            (all_results["model"] == "lstm") &
            (all_results["feature_set"] == feature_set)
        ]
        .sort_values("seed")
    )

    attention = (
        all_results[
            (all_results["model"] == "attention_lstm") &
            (all_results["feature_set"] == feature_set)
        ]
        .sort_values("seed")
    )

    if len(lstm) != N_RUNS or len(attention) != N_RUNS:
        print(f"\nSkipping {feature_set}: missing runs.")
        continue

    lstm_rmse = lstm["RMSE"].values
    attention_rmse = attention["RMSE"].values

    rmse_improvements = lstm_rmse - attention_rmse

    t_stat, t_p = ttest_rel(lstm_rmse, attention_rmse)

    try:
        w_stat, w_p = wilcoxon(lstm_rmse, attention_rmse)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    significance_rows.append({
        "feature_set": feature_set,
        "lstm_rmse_mean": np.mean(lstm_rmse),
        "lstm_rmse_std": np.std(lstm_rmse, ddof=1),
        "attention_rmse_mean": np.mean(attention_rmse),
        "attention_rmse_std": np.std(attention_rmse, ddof=1),
        "mean_rmse_improvement_lstm_minus_attention": np.mean(rmse_improvements),
        "paired_ttest_p_value": t_p,
        "wilcoxon_p_value": w_p,
        "significant_ttest_0.05": t_p < 0.05,
        "significant_wilcoxon_0.05": w_p < 0.05 if not np.isnan(w_p) else False,
    })

    print(f"\nFeature set: {feature_set}")
    print(f"LSTM RMSE:          {np.mean(lstm_rmse):.3f} ± {np.std(lstm_rmse, ddof=1):.3f}")
    print(f"Attention RMSE:     {np.mean(attention_rmse):.3f} ± {np.std(attention_rmse, ddof=1):.3f}")
    print(f"Mean improvement:   {np.mean(rmse_improvements):.3f}")
    print(f"Paired t-test p:    {t_p:.6f}")

    if not np.isnan(w_p):
        print(f"Wilcoxon p:         {w_p:.6f}")
    else:
        print("Wilcoxon p:         NA")

    if t_p < 0.05:
        print("Result: statistically significant by paired t-test")
    else:
        print("Result: NOT statistically significant by paired t-test")


significance_df = pd.DataFrame(significance_rows)

significance_df.to_csv(
    os.path.join(FINAL_OUTPUT_DIR, "statistical_tests_lstm_vs_attention.csv"),
    index=False
)


# ============================================================
# LATEX TABLE
# ============================================================

latex_table = final_table.to_latex(index=False, escape=False)

with open(
    os.path.join(FINAL_OUTPUT_DIR, "final_results_mean_std_latex.tex"),
    "w"
) as f:
    f.write(latex_table)


print("\nSaved outputs in:")
print(FINAL_OUTPUT_DIR)
print("\nFiles created:")
print("- all_runs_raw.csv")
print("- final_results_mean_std.csv")
print("- final_results_numeric.csv")
print("- statistical_tests_lstm_vs_attention.csv")
print("- final_results_mean_std_latex.tex")
print("\nDONE.")
