from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class DataProcessor:
    """
    - Load minute-level frame (kept for output).
    - Build hourly numeric view; interpolate + smooth.
    - Robust hour-of-day normalization using TRAINING med/IQR.
    - Select non-constant base features; winsorize by training quantiles.
    - Augment (diff + rolling std) WITHOUT dropping early rows.
    - Robust-scale (fit on training).
    - Broadcast hourly results back to minutes, adding ONLY the 8 required columns.
    """

    def __init__(self, smoothing_window: int = 5, resample_freq: str = "h"):
        self.smoothing_window = int(max(1, smoothing_window))
        self.resample_freq = resample_freq

    # ---------- load & hourly view ----------

    def load_raw(self, file_path: str) -> pd.DataFrame:
        print("--- load_raw: reading CSV ---")
        df = pd.read_csv(file_path)
        if "Time" not in df.columns:
            raise ValueError("Expected 'Time' column in input CSV.")
        df["Time"] = pd.to_datetime(df["Time"], format="mixed", errors="coerce")
        if df["Time"].isna().any():
            raise ValueError(f"Found {int(df['Time'].isna().sum())} invalid timestamps.")
        df = df.sort_values("Time").drop_duplicates(subset=["Time"]).set_index("Time")
        print(f"Raw frame: shape={df.shape} (this cadence will be preserved in output)")
        return df

    def hourly_numeric_view(self, original_df: pd.DataFrame) -> pd.DataFrame:
        num = original_df.select_dtypes(include=[np.number]).copy()
        if self.resample_freq:
            num = num.resample(self.resample_freq).mean()
        num = num.interpolate(method="time").ffill().bfill()
        if self.smoothing_window > 1:
            num = num.rolling(window=self.smoothing_window, min_periods=1).mean()
        num.index.name = "Time"
        print(f"Hourly modeling view: shape={num.shape}, numeric_cols={num.shape[1]}")
        return num

    # ---------- diurnal normalization ----------

    @staticmethod
    def hod_normalize(full_hourly: pd.DataFrame, train_hourly: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
        g = train_hourly.copy()
        g["_hod_"] = g.index.hour
        med = g.groupby("_hod_").median(numeric_only=True).drop(columns="_hod_", errors="ignore")
        q75 = g.groupby("_hod_").quantile(0.75, numeric_only=True)
        q25 = g.groupby("_hod_").quantile(0.25, numeric_only=True)
        iqr = (q75 - q25).reindex_like(med)

        global_med = train_hourly.median(numeric_only=True)
        global_iqr = (train_hourly.quantile(0.75) - train_hourly.quantile(0.25)).replace(0.0, eps)
        for h in range(24):
            if h not in med.index:
                med.loc[h] = global_med
                iqr.loc[h] = global_iqr

        out = full_hourly.copy()
        hours = out.index.hour
        for h in range(24):
            mask = hours == h
            if not np.any(mask):
                continue
            m = med.loc[h]
            s = iqr.loc[h].replace(0.0, eps)
            out.loc[mask, :] = (out.loc[mask, :] - m) / s
        out.index.name = "Time"
        print("Applied robust hour-of-day normalization using training med/IQR.")
        return out

    # ---------- training window ----------

    @staticmethod
    def get_training_data(df_hourly: pd.DataFrame) -> pd.DataFrame:
        start = "2004-01-01 00:00:00"
        end   = "2004-01-06 00:00:00"   # exclusive
        sl = df_hourly[(df_hourly.index >= start) & (df_hourly.index < end)]
        print(f"Training slice (hourly): shape={sl.shape}")
        return sl

    # ---------- base selection & clipping ----------

    @staticmethod
    def select_and_clip_base(train_hr: pd.DataFrame, full_hr: pd.DataFrame, q_low: float, q_high: float):
        tr = train_hr.copy()
        fl = full_hr.copy()
        std = tr.std()
        iqr = tr.quantile(0.75) - tr.quantile(0.25)
        keep = std[(std > 1e-9) & (iqr > 1e-9)].index.tolist()
        if not keep:
            raise ValueError("All numeric features are (near) constant in training.")

        ql = tr[keep].quantile(q_low)
        qh = tr[keep].quantile(q_high)
        tr_clip = tr[keep].clip(lower=ql, upper=qh, axis="columns")
        fl_clip = fl[keep].clip(lower=ql, upper=qh, axis="columns")
        print(f"Kept {len(keep)} base features. Winsorized at [{q_low*100:.1f}%, {q_high*100:.1f}%] based on training.")
        return tr_clip, fl_clip, keep

    # ---------- augmentation & scaling ----------

    @staticmethod
    def build_augmented(train_base: pd.DataFrame, full_base: pd.DataFrame, roll_window: int):
        def augment(df: pd.DataFrame):
            dif1 = df.diff().fillna(0.0).add_suffix("__diff1")
            rstd = df.rolling(roll_window, min_periods=1).std(ddof=0).fillna(0.0).add_suffix(f"__rstd{roll_window}")
            aug = pd.concat([df, dif1, rstd], axis=1)
            mapping = {c: c.split("__")[0] for c in aug.columns}
            return aug, mapping

        train_aug, m = augment(train_base)
        full_aug, _ = augment(full_base)
        return train_aug, full_aug, m

    @staticmethod
    def robust_scale(train_aug: pd.DataFrame, full_aug: pd.DataFrame):
        scaler = RobustScaler().fit(train_aug)
        X_train = pd.DataFrame(scaler.transform(train_aug), index=train_aug.index, columns=train_aug.columns)
        X_full  = pd.DataFrame(scaler.transform(full_aug),  index=full_aug.index,  columns=full_aug.columns)
        return X_train, X_full

    # ---------- save (EXACTLY 8 new columns) ----------

    @staticmethod
    def save_results_broadcast(original_df: pd.DataFrame, hourly_results: pd.DataFrame, output_path: str) -> None:
        """
        Adds EXACTLY these 8 new columns to the original cadence:
          - abnormality_score
          - top_feature_1 .. top_feature_7
        """
        print(f"Saving results (broadcast hourly -> original cadence) -> {output_path}")
        out = original_df.copy()
        out["__HourKey__"] = out.index.floor("h")

        hr = hourly_results.copy()
        hr["__HourKey__"] = hr.index
        keep_cols = ["abnormality_score"] + [f"top_feature_{i}" for i in range(1, 8)]
        keep_cols = [c for c in keep_cols if c in hr.columns]

        merged = out.merge(hr[["__HourKey__"] + keep_cols], how="left", on="__HourKey__").drop(columns="__HourKey__")

        # Fill any residual gaps to avoid blanks
        merged["abnormality_score"] = merged["abnormality_score"].fillna(0.0)
        for c in [f"top_feature_{i}" for i in range(1, 8)]:
            if c in merged.columns:
                merged[c] = merged[c].fillna("")

        merged.reset_index().to_csv(output_path, index=False)
        print("Saved.")
