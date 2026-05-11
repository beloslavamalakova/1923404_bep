# BEP LSTM Ablation Study Code

This folder contains code for the ablation study:

1. **glucose only**
2. **glucose + insulin**: basal, bolus, bolus over last 30 min, time since last bolus
3. **glucose + meal**: carbs, carbs over last 30 min, time since last meal

Each feature set is tested on:

1. LSTM
2. LSTM + attention over timesteps

## 1. Install requirements

```bash
pip install pandas numpy scikit-learn torch lxml tqdm
```

## 2. Build features from XML

Run this from the same folder where your `data/` folder is located:

```bash
python build_features_ablation.py --base-path data
```

This creates:

```text
train_features_ablation.csv
test_features_ablation.csv
```

## 3. Run the ablation study

```bash
python run_lstm_attention_ablation.py --epochs 30
```

Outputs are saved in:

```text
ablation_outputs/ablation_results.csv
ablation_outputs/predictions_lstm_glucose_only.csv
ablation_outputs/predictions_attention_lstm_glucose_only.csv
...
ablation_outputs/attention_weights_glucose_only.csv
ablation_outputs/attention_weights_glucose_insulin.csv
ablation_outputs/attention_weights_glucose_meal.csv
```

## 4. Run on existing feature files

If you already have feature files and want to use them directly:

```bash
python run_lstm_attention_ablation.py \
  --train-file train_features.csv \
  --test-file test_features.csv \
  --epochs 30
```

For the meal ablation, your feature files must contain meal columns such as `carbs`, `carbs_30min`, and `time_since_last_meal_min`. If not, run `build_features_ablation.py` first.
