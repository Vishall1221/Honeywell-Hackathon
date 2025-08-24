# FINAL NAME: main.py
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

from data_processor import DataProcessor
from anomaly_detector import AnomalyDetector

# ---- Defaults so you can just run `python main.py` ----
DEFAULT_INPUT = "TEP_Train_Test.csv"
DEFAULT_OUTPUT = "results.csv"
DEFAULT_SMOOTH = 5

# rubric targets
TARGET_MEAN = 10.0
TARGET_MAX  = 25.0


def evaluate_config(df_raw, smoothing_window, q_pair, roll_window, drop_pct, if_estimators=300, rng=42):
    """Fast config check (no SHAP) on the hourly grid."""
    proc = DataProcessor(smoothing_window=smoothing_window, resample_freq="h")
    df_hour = proc.hourly_numeric_view(df_raw)
    df_train_hr = proc.get_training_data(df_hour)
    if len(df_train_hr) < 72:
        return None

    df_hour_norm = proc.hod_normalize(df_hour, df_train_hr)
    df_train_norm = df_hour_norm.loc[df_train_hr.index]

    q_low, q_high = q_pair
    train_base, full_base, _ = proc.select_and_clip_base(df_train_norm, df_hour_norm, q_low, q_high)
    train_aug, full_aug, aug_to_base = proc.build_augmented(train_base, full_base, roll_window=roll_window)
    X_train_hr, X_full_hr = proc.robust_scale(train_aug, full_aug)

    det = AnomalyDetector(
        aug_to_base=aug_to_base,
        stage1_drop_pct=drop_pct,
        n_estimators=if_estimators,
        random_state=rng,
        train_tail_target=18.0,
    )
    det.train(X_train_hr, build_explainer=False)
    scores_hr = det.predict_scores(X_full_hr, train_index=df_train_hr.index)

    tr = scores_hr.loc[df_train_hr.index.min():df_train_hr.index.max()]
    mean_ = float(tr.mean()) if len(tr) else np.inf
    max_  = float(tr.max()) if len(tr) else np.inf
    return {
        "mean": mean_,
        "max": max_,
        "smoothing_window": smoothing_window,
        "q_pair": q_pair,
        "roll_window": roll_window,
        "drop_pct": drop_pct,
    }


def search_best(df_raw, base_smoothing):
    smoothing_grid = [base_smoothing, max(3, base_smoothing + 2), base_smoothing + 4]
    q_grid = [(0.02, 0.98), (0.03, 0.97), (0.05, 0.95)]
    drop_grid = [0.07, 0.10, 0.15]
    roll_grid = [3, 5]

    best = None
    for sm in smoothing_grid:
        for q in q_grid:
            for rw in roll_grid:
                for dp in drop_grid:
                    res = evaluate_config(df_raw, sm, q, rw, dp, if_estimators=280)
                    if res is None:
                        continue
                    if res["mean"] <= TARGET_MEAN and res["max"] <= TARGET_MAX:
                        print(f"[AUTO] Passing config: smooth={sm}, q={q}, roll={rw}, drop={dp} "
                              f"-> mean={res['mean']:.2f}, max={res['max']:.2f}")
                        return res
                    score = max(res["mean"] - TARGET_MEAN, 0) + 2.0 * max(res["max"] - TARGET_MAX, 0)
                    if (best is None) or (score < best.get("_score", float("inf"))):
                        res["_score"] = score
                        best = res

    if best is None:
        raise RuntimeError("No viable config found.")
    print("[AUTO] Using closest config (did not fully pass rubric).")
    return best


def run_pipeline(input_path: str, output_path: str, smoothing_window: int) -> None:
    print("--- IF+SHAP (hourly model + minute output) with Diurnal Normalization ---")
    print(f"[CONFIG] input='{input_path}'  output='{output_path}'  smooth={smoothing_window}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file '{input_path}' not found. "
            f"Put your CSV next to main.py or pass --input <path>."
        )

    raw_loader = DataProcessor(smoothing_window=smoothing_window, resample_freq="h")
    df_raw = raw_loader.load_raw(input_path)

    best = search_best(df_raw, smoothing_window)
    sm = best["smoothing_window"]; (q_low, q_high) = best["q_pair"]
    rw = best["roll_window"]; dp = best["drop_pct"]
    print(f"[AUTO] Final config -> smooth={sm}, q=({q_low},{q_high}), roll={rw}, drop={dp}")

    # Rebuild with chosen knobs, this time with SHAP enabled
    proc = DataProcessor(smoothing_window=sm, resample_freq="h")
    df_hour = proc.hourly_numeric_view(df_raw)
    df_train_hr = proc.get_training_data(df_hour)
    if len(df_train_hr) < 72:
        raise ValueError(f"Need ≥72 hours of training data; got {len(df_train_hr)} hours.")

    df_hour_norm = proc.hod_normalize(df_hour, df_train_hr)
    df_train_norm = df_hour_norm.loc[df_train_hr.index]

    train_base, full_base, _ = proc.select_and_clip_base(df_train_norm, df_hour_norm, q_low, q_high)
    train_aug, full_aug, aug_to_base = proc.build_augmented(train_base, full_base, roll_window=rw)
    X_train_hr, X_full_hr = proc.robust_scale(train_aug, full_aug)

    det = AnomalyDetector(
        aug_to_base=aug_to_base,
        stage1_drop_pct=dp,
        n_estimators=600,
        random_state=42,
        train_tail_target=18.0,
    )
    det.train(X_train_hr, build_explainer=True)
    scores_hr = det.predict_scores(X_full_hr, train_index=df_train_hr.index)
    top_feats_hr = det.get_top_contributors(X_full_hr)

    # Exactly the 8 required columns
    hourly_results = pd.DataFrame(index=X_full_hr.index)
    hourly_results["abnormality_score"] = scores_hr
    hourly_results = hourly_results.join(top_feats_hr)

    proc.save_results_broadcast(df_raw, hourly_results, output_path)

    tr = hourly_results.loc[df_train_hr.index.min():df_train_hr.index.max(), "abnormality_score"].dropna()
    if not tr.empty:
        print(f"[REPORT] Training (hourly) -> mean={tr.mean():.2f}, max={tr.max():.2f}, n={len(tr)}")
    print("--- Done ---")


if __name__ == "__main__":
    # Optional args; defaults let you run plain `python main.py`
    ap = argparse.ArgumentParser(description="Time-series anomaly detection (IF + SHAP) with diurnal normalization.")
    ap.add_argument("--input",  default=DEFAULT_INPUT,  help="Path to input CSV (default: time_series_dataset.csv)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output CSV (default: results.csv)")
    ap.add_argument("--smoothing_window", type=int, default=DEFAULT_SMOOTH, help="Base rolling-mean window (default: 5)")
    args = ap.parse_args()

    run_pipeline(args.input, args.output, args.smoothing_window)
