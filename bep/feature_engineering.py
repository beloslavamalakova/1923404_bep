import pandas as pd


def add_features(df):

    df = df.sort_values(["patient", "timestamp"]).copy()

    # --------------------------------------------------
    # GLUCOSE LAGS (60 min history)
    # --------------------------------------------------
    for lag in range(1, 13):
        df[f"glucose_lag_{lag}"] = (
            df.groupby("patient")["glucose"].shift(lag)
        )

    # --------------------------------------------------
    # 30-MIN FUTURE TARGET
    # --------------------------------------------------
    df["target"] = (
        df.groupby("patient")["glucose"].shift(-6)
    )

    # --------------------------------------------------
    # BOLUS: cumulative last 30 min
    # --------------------------------------------------
    df["bolus_30min"] = (
        df.groupby("patient")["bolus"]
        .rolling(window=6, min_periods=1)
        .sum()
        .reset_index(0, drop=True)
    )

    # --------------------------------------------------
    # OPTIONAL: glucose slope
    # --------------------------------------------------
    df["glucose_slope"] = (
        df.groupby("patient")["glucose"]
        .diff()
    )

    df = df.dropna()

    return df


if __name__ == "__main__":

    train = pd.read_csv("model_train.csv", parse_dates=["timestamp"])
    test = pd.read_csv("model_test.csv", parse_dates=["timestamp"])

    train_feat = add_features(train)
    test_feat = add_features(test)

    train_feat.to_csv("train_features.csv", index=False)
    test_feat.to_csv("test_features.csv", index=False)

    print("Saved train_features.csv and test_features.csv")
