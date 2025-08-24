from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import shap


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-9)


class AnomalyDetector:
    """
    Isolation Forest + SHAP with:
      - stage-1 cleaning of the training window (drop top X% outliers)
      - percentile-calibrated score (0..100), anchored to training-window max
      - SHAP feature aggregation back to base feature names
    """

    def __init__(
        self,
        aug_to_base: dict[str, str] | None,
        stage1_drop_pct: float = 0.07,
        n_estimators: int = 600,
        random_state: int = 42,
        train_tail_target: float = 18.0,  # push training-window max under ~20
        gamma_cap: float = 40.0,
        jitter_max: float = 0.005,
    ):
        self.aug_to_base = aug_to_base or {}
        self.stage1_drop_pct = float(stage1_drop_pct)
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=256,
            contamination="auto",
            n_jobs=-1,
            random_state=random_state,
        )
        self.feature_names_: list[str] | None = None
        self.explainer: shap.TreeExplainer | None = None

        self.train_raw_med_: float | None = None
        self.train_raw_mad_: float | None = None

        self._train_tail_target = float(train_tail_target)
        self._gamma_cap = float(gamma_cap)
        self._jitter_max = float(jitter_max)

    # ---------- helpers ----------

    @staticmethod
    def _percentile_0_100(x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n <= 1:
            return np.zeros_like(x, dtype=float)
        order = np.argsort(x)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        return 100.0 * (ranks / (n - 1))

    @staticmethod
    def _stage1_filter(X_train: pd.DataFrame, drop_pct: float, random_state: int = 123) -> pd.DataFrame:
        if len(X_train) < 10 or drop_pct <= 0.0:
            return X_train
        stage1 = IsolationForest(
            n_estimators=250,
            max_samples=min(len(X_train), 512),
            contamination="auto",
            n_jobs=-1,
            random_state=random_state,
        )
        stage1.fit(X_train)
        raw = -stage1.score_samples(X_train)
        thr = np.percentile(raw, 100 * (1.0 - drop_pct))
        keep = raw <= thr
        print(f"[Two-stage] Dropping top {drop_pct*100:.1f}% training hours ({len(X_train)-int(keep.sum())} / {len(X_train)}).")
        return X_train.loc[keep]

    # ---------- training ----------

    def train(self, X_train_aug: pd.DataFrame, build_explainer: bool = True) -> None:
        if X_train_aug.empty:
            raise ValueError("Training data is empty.")
        self.feature_names_ = list(X_train_aug.columns)

        cleaned = self._stage1_filter(X_train_aug, self.stage1_drop_pct)
        self.model.set_params(max_samples=min(len(cleaned), 512))
        self.model.fit(cleaned)

        # Baseline statistics for "all-normal" guard
        train_raw = -self.model.score_samples(cleaned)
        self.train_raw_med_ = float(np.median(train_raw))
        self.train_raw_mad_ = _mad(train_raw)

        if build_explainer:
            # small background set for TreeExplainer stability
            bg_n = min(300, len(cleaned))
            bg = cleaned.sample(n=bg_n, random_state=42) if bg_n > 0 else cleaned
            self.explainer = shap.TreeExplainer(self.model, data=bg)
        else:
            self.explainer = None

    # ---------- scoring ----------

    def _gamma_from_train_max(self, raw_full: np.ndarray, full_index: pd.Index, train_index: pd.Index) -> float:
        mask = pd.Index(full_index).isin(pd.Index(train_index))
        if not np.any(mask):
            return 1.0
        train_max_raw = float(np.max(raw_full[mask]))
        p = (np.sum(raw_full <= train_max_raw) - 1e-9) / max(len(raw_full) - 1, 1)
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        eff_target = max(self._train_tail_target - 0.2, 0.1) / 100.0
        gamma = np.log(eff_target) / np.log(p)
        gamma = float(np.clip(gamma, 1.0, self._gamma_cap))
        print(f"[Calibrate] train_window_max_p={p:.4f} -> target={self._train_tail_target:.2f} -> gamma={gamma:.3f}")
        return gamma

    def predict_scores(self, X_full_aug: pd.DataFrame, train_index: pd.Index | None = None) -> pd.Series:
        raw = -self.model.score_samples(X_full_aug)         # higher = more anomalous
        raw_pct = self._percentile_0_100(raw)               # 0..100 global percentile

        # "all-normal" guard: keep small yet non-zero values
        if self.train_raw_med_ is not None and self.train_raw_mad_ is not None:
            if raw.max() <= self.train_raw_med_ + 3.0 * self.train_raw_mad_:
                rng = np.random.default_rng(42)
                cal = 0.2 * raw_pct + rng.uniform(0, self._jitter_max, size=len(raw_pct))
                return pd.Series(np.clip(cal, 0, 100), index=X_full_aug.index, name="abnormality_score")

        gamma = self._gamma_from_train_max(raw, X_full_aug.index, train_index) if train_index is not None else 1.0
        p = np.clip(raw_pct / 100.0, 0.0, 1.0)
        cal = (p ** gamma) * 100.0
        rng = np.random.default_rng(42)
        cal = np.clip(cal + rng.uniform(0, self._jitter_max, size=len(cal)), 0, 100)
        return pd.Series(cal, index=X_full_aug.index, name="abnormality_score")

    # ---------- SHAP top contributors ----------

    def get_top_contributors(self, X_full_aug: pd.DataFrame) -> pd.DataFrame:
        if self.explainer is None or self.feature_names_ is None:
            raise RuntimeError("Explainer not initialized. Call train() with build_explainer=True.")

        shap_vals = self.explainer.shap_values(X_full_aug)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        abs_shap = np.abs(shap_vals)

        # aggregate augmented -> base names
        aug_cols = list(X_full_aug.columns)
        base_names = sorted(set(self.aug_to_base.get(c, c) for c in aug_cols))
        base_index = {b: i for i, b in enumerate(base_names)}
        grouped = np.zeros((abs_shap.shape[0], len(base_names)), dtype=float)

        for j, aug in enumerate(aug_cols):
            b = self.aug_to_base.get(aug, aug)
            grouped[:, base_index[b]] += abs_shap[:, j]

        row_sum = grouped.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        pct = grouped / row_sum * 100.0

        rows = []
        for i in range(pct.shape[0]):
            pairs = [(base_names[j], pct[i, j], grouped[i, j]) for j in range(len(base_names)) if pct[i, j] >= 1.0]
            pairs.sort(key=lambda x: (-x[2], x[0]))  # by magnitude desc, then alphabetical
            top = [name for name, _, _ in pairs[:7]] + [""] * (7 - min(7, len(pairs)))
            rows.append(top)

        cols = [f"top_feature_{k}" for k in range(1, 8)]
        return pd.DataFrame(rows, columns=cols, index=X_full_aug.index)
