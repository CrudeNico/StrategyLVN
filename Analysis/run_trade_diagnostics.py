#!/usr/bin/env python3
"""Trade diagnostics pipeline with feature extraction and explainability."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from pyod.models.iforest import IForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
from xgboost import XGBClassifier

from dowhy import CausalModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FINDINGS_RESULTS = PROJECT_ROOT / "Findings" / "Results"
BACKTEST_RESULTS = PROJECT_ROOT / "Backtest" / "Results"
OUTPUT_DIR = PROJECT_ROOT / "Analysis" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_FILE = DATA_DIR / "1minute.txt"
EMA_FILE = FINDINGS_RESULTS / "ema20.csv"
TRADE_RESULTS_FILE = BACKTEST_RESULTS / "trade_results.csv"
LVN_FILE = FINDINGS_RESULTS / "zigzag_lvn_summary.csv"

WINDOW_MINUTES = 30
MIN_WINDOW_POINTS = 10
RANDOM_STATE = 42


def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(
        PRICE_FILE,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={
            "date": "string",
            "time": "string",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M")
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df


def compute_atr(price_df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = price_df["high"] - price_df["low"]
    high_close = (price_df["high"] - price_df["close"].shift()).abs()
    low_close = (price_df["low"] - price_df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def compute_volatility(price_df: pd.DataFrame, window: int = 30) -> pd.Series:
    returns = price_df["close"].pct_change()
    vol = returns.rolling(window=window, min_periods=1).std()
    return vol


def load_ema_series() -> pd.Series:
    ema_df = pd.read_csv(EMA_FILE, parse_dates=["datetime"])
    ema_df.sort_values("datetime", inplace=True)
    ema_cols = [col for col in ema_df.columns if col.startswith("ema_")]
    if not ema_cols:
        raise ValueError("EMA file does not contain an ema_* column")
    ema_series = ema_df.set_index("datetime")[ema_cols[0]].astype(float)
    ema_series.name = "ema_value"
    return ema_series


def load_trades() -> pd.DataFrame:
    trades = pd.read_csv(TRADE_RESULTS_FILE, parse_dates=["entry_time", "exit_time", "target_time"])
    trades.sort_values("entry_time", inplace=True)
    return trades


def load_lvn() -> pd.DataFrame:
    return pd.read_csv(
        LVN_FILE,
        parse_dates=["from_datetime", "to_datetime"],
    )


def build_base_features(trades: pd.DataFrame, price_df: pd.DataFrame, ema_series: pd.Series) -> pd.DataFrame:
    features = trades.copy()
    features["win"] = (features["pnl_eur"] > 0).astype(int)
    features["ema_value"] = ema_series.reindex(features["entry_time"]).to_numpy()
    features["dist_to_ema"] = features["entry_price"] - features["ema_value"]
    features["stop_distance"] = (features["entry_price"] - features["stop_price"]).abs()
    features["reward_distance"] = features["target_price"] - features["entry_price"]
    features["lvn_band_width"] = features["lvn_price_max"] - features["lvn_price_min"]
    features["entry_hour"] = features["entry_time"].dt.hour
    features["entry_weekday"] = features["entry_time"].dt.weekday
    features["session"] = np.where(
        (features["entry_hour"] >= 9) & (features["entry_hour"] < 16), "RTH", "OFF"
    )
    features["holding_minutes"] = features["holding_minutes"].fillna(0.0)

    atr = compute_atr(price_df)
    vol = compute_volatility(price_df)
    features["atr"] = atr.reindex(features["entry_time"]).to_numpy()
    features["volatility"] = vol.reindex(features["entry_time"]).to_numpy()
    features["volume_at_entry"] = price_df["volume"].reindex(features["entry_time"]).to_numpy()

    return features


def extract_tsfresh_features(features: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_id, entry_time in zip(features["trade_id"], features["entry_time"]):
        window = price_df.loc[
            entry_time - pd.Timedelta(minutes=WINDOW_MINUTES) : entry_time
        ].copy()
        if window.empty or len(window) < MIN_WINDOW_POINTS:
            continue
        temp = window[["close"]].reset_index()
        temp["id"] = trade_id
        temp.rename(columns={"datetime": "time", "close": "value"}, inplace=True)
        rows.append(temp)

    if not rows:
        return pd.DataFrame()

    time_series = pd.concat(rows, ignore_index=True)
    extracted = extract_features(
        time_series,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True,
        n_jobs=0,
    )
    extracted.reset_index(inplace=True)
    extracted.rename(columns={"index": "trade_id"}, inplace=True)
    return extracted


def prepare_feature_matrix(features: pd.DataFrame, tsfresh_df: pd.DataFrame) -> pd.DataFrame:
    merged = features.merge(tsfresh_df, on="trade_id", how="left") if not tsfresh_df.empty else features
    merged = pd.get_dummies(merged, columns=["direction", "session"], drop_first=True)
    drop_cols = {
        "entry_time",
        "exit_time",
        "target_time",
        "exit_reason",
        "pnl_eur",
        "risk_eur",
        "reward_eur",
        "units",
        "balance_before_trade",
        "balance_after_trade",
        "target_leg_id",
        "stop_leg_id",
        "from_leg_id",
        "to_leg_id",
        "transition_type",
    }
    merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)
    merged.dropna(inplace=True)
    return merged


def train_model(feature_df: pd.DataFrame):
    trade_ids = feature_df["trade_id"].copy()
    y = feature_df.pop("win")
    feature_df = feature_df.drop(columns=["trade_id"], errors="ignore")
    X = feature_df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    metrics["roc_auc"] = roc_auc

    metrics_path = OUTPUT_DIR / "model_metrics.json"
    with metrics_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)

    return model, X, y, trade_ids


def generate_shap_outputs(model: XGBClassifier, X: pd.DataFrame, trades: pd.DataFrame) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=200)
    plt.close()

    worst_idx = trades["pnl_eur"].idxmin()
    if worst_idx in X.index:
        shap_explanation = explainer(X.loc[[worst_idx]])
        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(shap_explanation[0], max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"shap_waterfall_trade_{int(trades.loc[worst_idx, 'trade_id'])}.png", dpi=200)
        plt.close()

        lime_explainer = LimeTabularExplainer(
            X.values,
            feature_names=X.columns.tolist(),
            class_names=["Loss", "Win"],
            verbose=False,
            mode="classification",
            random_state=RANDOM_STATE,
        )
        exp = lime_explainer.explain_instance(
            X.loc[worst_idx].values,
            model.predict_proba,
            num_features=10,
        )
        exp.save_to_file(str(OUTPUT_DIR / f"lime_trade_{int(trades.loc[worst_idx, 'trade_id'])}.html"))


def run_anomaly_detection(X: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    detector = IForest(contamination=0.1, random_state=RANDOM_STATE)
    detector.fit(X_scaled)
    flags = detector.predict(X_scaled)
    trades["anomaly_flag"] = flags
    trades.to_csv(OUTPUT_DIR / "trade_features.csv", index=False)
    return trades["anomaly_flag"]


def run_causal_analysis(feature_df: pd.DataFrame) -> None:
    causal_df = feature_df.copy()
    causal_df["dist_to_ema_abs"] = causal_df["dist_to_ema"].abs()
    causal_df = causal_df[[
        "win",
        "dist_to_ema_abs",
        "stop_distance",
        "reward_distance",
        "lvn_band_width",
        "atr",
        "volatility",
        "entry_hour",
    ]].dropna()

    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("dist_to_ema_abs", "win"),
            ("stop_distance", "win"),
            ("stop_distance", "dist_to_ema_abs"),
            ("reward_distance", "win"),
            ("reward_distance", "dist_to_ema_abs"),
            ("lvn_band_width", "win"),
            ("lvn_band_width", "dist_to_ema_abs"),
            ("atr", "win"),
            ("atr", "dist_to_ema_abs"),
            ("volatility", "win"),
            ("volatility", "dist_to_ema_abs"),
            ("entry_hour", "win"),
            ("entry_hour", "dist_to_ema_abs"),
        ]
    )

    model = CausalModel(
        data=causal_df,
        treatment="dist_to_ema_abs",
        outcome="win",
        graph=graph,
    )

    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )
    refutation = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="bootstrap_refuter",
        number_of_simulations=100,
    )

    with (OUTPUT_DIR / "causal_summary.txt").open("w") as fh:
        fh.write("Identified estimand:\n")
        fh.write(str(identified_estimand))
        fh.write("\n\nEstimate:\n")
        fh.write(str(estimate))
        fh.write("\n\nRefutation:\n")
        fh.write(str(refutation))


def main() -> None:
    price_df = load_price_data()
    ema_series = load_ema_series()
    trades = load_trades()

    base_features = build_base_features(trades, price_df, ema_series)
    tsfresh_features = extract_tsfresh_features(base_features, price_df)
    feature_matrix = prepare_feature_matrix(base_features, tsfresh_features)

    model, X, y, trade_ids = train_model(feature_matrix.copy())
    trades_aligned = trades.loc[X.index].copy()
    trades_aligned["trade_id"] = trade_ids.loc[X.index]

    generate_shap_outputs(model, X, trades_aligned)
    run_anomaly_detection(X, trades_aligned)
    run_causal_analysis(feature_matrix.loc[X.index].assign(win=y.loc[X.index]))

    print(f"Diagnostics complete. Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
