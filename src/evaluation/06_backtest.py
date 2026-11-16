#!/usr/bin/env python3
"""06_backtest.py

Etapa 6: Avaliação e Backtesting com janela temporal expansiva (walk-forward validation).
Compara o modelo LightGBM com o Baseline Ingênuo Semanal em MAE e MAPE ao longo
de múltiplas janelas temporais no futuro, garantindo zero vazamento de dados.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

feat_mod = importlib.import_module("src.features.05_feature_engineering")
train_mod = importlib.import_module("src.models.05_train_lightgbm")

FEATURE_COLUMNS = feat_mod.FEATURE_COLUMNS
DEFAULT_LGBM_PARAMS = train_mod.DEFAULT_LGBM_PARAMS
DEFAULT_TRACKING_URI = train_mod.DEFAULT_TRACKING_URI
EXPERIMENT_NAME = train_mod.EXPERIMENT_NAME
compute_metrics = train_mod.compute_metrics
prepare_modeling_dataset = train_mod.prepare_modeling_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluation.backtest")


def generate_temporal_folds(
    df_model: pd.DataFrame,
    backtest_start: str = "2024-01-01",
    fold_days: int = 90,
    val_days: int = 90,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Gera janelas sequenciais (val_start, test_start, test_end) no tempo."""
    max_date = df_model["data_referencia"].max()
    cursor = pd.Timestamp(backtest_start)
    folds: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    while cursor <= max_date:
        test_start = cursor
        test_end = min(cursor + pd.Timedelta(days=fold_days - 1), max_date)
        val_start = test_start - pd.Timedelta(days=val_days)
        folds.append((val_start, test_start, test_end))
        cursor = test_end + pd.Timedelta(days=1)

    return folds


def run_walk_forward_backtest(
    warehouse_path: Path = Path("data/warehouse.duckdb"),
    reports_dir: Path = Path("reports"),
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    backtest_start: str = "2024-01-01",
    fold_days: int = 90,
    val_days: int = 90,
    log_mlflow: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Executa backtesting temporal expansivo comparando LightGBM vs Baseline."""
    df_model = prepare_modeling_dataset(warehouse_path=warehouse_path)
    folds = generate_temporal_folds(
        df_model,
        backtest_start=backtest_start,
        fold_days=fold_days,
        val_days=val_days,
    )

    fold_records: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []

    logger.info(
        "Iniciando backtesting temporal com %d janelas a partir de %s ...",
        len(folds),
        backtest_start,
    )

    for idx, (val_start, test_start, test_end) in enumerate(folds, start=1):
        train_df = df_model[df_model["data_referencia"] < val_start]
        val_df = df_model[
            (df_model["data_referencia"] >= val_start) & (df_model["data_referencia"] < test_start)
        ]
        test_df = df_model[
            (df_model["data_referencia"] >= test_start) & (df_model["data_referencia"] <= test_end)
        ]

        if train_df.empty or val_df.empty or test_df.empty:
            continue

        model = lgb.LGBMRegressor(**DEFAULT_LGBM_PARAMS)
        model.fit(
            train_df[FEATURE_COLUMNS],
            train_df["carga_energia_mwmed"],
            eval_X=val_df[FEATURE_COLUMNS],
            eval_y=val_df["carga_energia_mwmed"],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        y_true = test_df["carga_energia_mwmed"].to_numpy()
        y_pred_lgbm = model.predict(test_df[FEATURE_COLUMNS])
        y_pred_base = test_df["lag_7"].to_numpy()

        lgbm_m = compute_metrics(y_true, y_pred_lgbm)
        base_m = compute_metrics(y_true, y_pred_base)
        mae_gain = ((base_m["mae_mwmed"] - lgbm_m["mae_mwmed"]) / base_m["mae_mwmed"]) * 100.0
        mape_gain = ((base_m["mape_pct"] - lgbm_m["mape_pct"]) / base_m["mape_pct"]) * 100.0

        fold_records.append(
            {
                "fold": idx,
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "n_dias": len(test_df),
                "train_rows": len(train_df),
                "baseline_mae_mwmed": base_m["mae_mwmed"],
                "lgbm_mae_mwmed": lgbm_m["mae_mwmed"],
                "mae_gain_pct": mae_gain,
                "baseline_mape_pct": base_m["mape_pct"],
                "lgbm_mape_pct": lgbm_m["mape_pct"],
                "mape_gain_pct": mape_gain,
            }
        )

        fold_preds = pd.DataFrame(
            {
                "fold": idx,
                "data_referencia": test_df["data_referencia"].to_numpy(),
                "y_true": y_true,
                "y_pred_baseline": y_pred_base,
                "y_pred_lgbm": y_pred_lgbm,
            }
        )
        pred_frames.append(fold_preds)

    folds_df = pd.DataFrame(fold_records)
    preds_df = pd.concat(pred_frames, ignore_index=True)

    overall_lgbm = compute_metrics(
        preds_df["y_true"].to_numpy(),
        preds_df["y_pred_lgbm"].to_numpy(),
    )
    overall_base = compute_metrics(
        preds_df["y_true"].to_numpy(),
        preds_df["y_pred_baseline"].to_numpy(),
    )

    overall_mae_gain = (
        (overall_base["mae_mwmed"] - overall_lgbm["mae_mwmed"]) / overall_base["mae_mwmed"]
    ) * 100.0
    overall_mape_gain = (
        (overall_base["mape_pct"] - overall_lgbm["mape_pct"]) / overall_base["mape_pct"]
    ) * 100.0
    folds_won = int((folds_df["lgbm_mae_mwmed"] < folds_df["baseline_mae_mwmed"]).sum())

    summary = {
        "n_folds": int(len(folds_df)),
        "folds_lgbm_better": folds_won,
        "total_test_days": int(len(preds_df)),
        "baseline_mae_mwmed": overall_base["mae_mwmed"],
        "baseline_mape_pct": overall_base["mape_pct"],
        "baseline_rmse_mwmed": overall_base["rmse_mwmed"],
        "lgbm_mae_mwmed": overall_lgbm["mae_mwmed"],
        "lgbm_mape_pct": overall_lgbm["mape_pct"],
        "lgbm_rmse_mwmed": overall_lgbm["rmse_mwmed"],
        "mae_gain_pct": overall_mae_gain,
        "mape_gain_pct": overall_mape_gain,
        "mae_reduction_mwmed": overall_base["mae_mwmed"] - overall_lgbm["mae_mwmed"],
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    folds_csv = reports_dir / "06_backtest_folds.csv"
    preds_csv = reports_dir / "06_backtest_predictions.csv"
    chart_png = reports_dir / "06_backtest_comparison.png"

    folds_df.to_csv(folds_csv, index=False)
    preds_df.to_csv(preds_csv, index=False)
    _save_backtest_chart(folds_df, preds_df, chart_png)

    if log_mlflow:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="backtest_walk_forward"):
            mlflow.log_params(
                {
                    **DEFAULT_LGBM_PARAMS,
                    "backtest_start": backtest_start,
                    "fold_days": fold_days,
                    "val_days": val_days,
                    "n_folds": summary["n_folds"],
                }
            )
            mlflow.log_metrics(
                {
                    "bt_baseline_mae_mwmed": summary["baseline_mae_mwmed"],
                    "bt_baseline_mape_pct": summary["baseline_mape_pct"],
                    "bt_lgbm_mae_mwmed": summary["lgbm_mae_mwmed"],
                    "bt_lgbm_mape_pct": summary["lgbm_mape_pct"],
                    "bt_mae_gain_pct": summary["mae_gain_pct"],
                    "bt_mape_gain_pct": summary["mape_gain_pct"],
                    "bt_folds_lgbm_better": summary["folds_lgbm_better"],
                }
            )
            mlflow.log_artifact(str(folds_csv))
            mlflow.log_artifact(str(chart_png))

    print("\n=== ETAPA 6: BACKTESTING TEMPORAL WALK-FORWARD (SUBSISTEMA SE) ===")
    print(
        f"Período avaliado: {preds_df['data_referencia'].min().date()} a "
        f"{preds_df['data_referencia'].max().date()} "
        f"({summary['total_test_days']} dias em {summary['n_folds']} janelas)"
    )
    print(f"Janelas em que LightGBM superou o Baseline: {folds_won}/{summary['n_folds']}")
    print("\n--- Desempenho por Janela Temporal ---")
    print(
        folds_df[
            [
                "fold",
                "test_start",
                "test_end",
                "baseline_mae_mwmed",
                "lgbm_mae_mwmed",
                "mae_gain_pct",
                "baseline_mape_pct",
                "lgbm_mape_pct",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )
    print("\n--- Tabela Comparativa Consolidada (Out-of-Sample) ---")
    print(
        f"Baseline Ingênuo (Semanal) : MAE = {summary['baseline_mae_mwmed']:.2f} MWmed | "
        f"MAPE = {summary['baseline_mape_pct']:.2f}% | "
        f"RMSE = {summary['baseline_rmse_mwmed']:.2f} MWmed"
    )
    print(
        f"LightGBM (Horizonte D+7)   : MAE = {summary['lgbm_mae_mwmed']:.2f} MWmed | "
        f"MAPE = {summary['lgbm_mape_pct']:.2f}% | "
        f"RMSE = {summary['lgbm_rmse_mwmed']:.2f} MWmed"
    )
    print(
        f"Ganho sobre Baseline       : {summary['mae_gain_pct']:+.2f}% em MAE "
        f"(-{summary['mae_reduction_mwmed']:.2f} MWmed) | "
        f"{summary['mape_gain_pct']:+.2f}% em MAPE"
    )
    print("====================================================================\n")

    return folds_df, preds_df, summary


def _save_backtest_chart(
    folds_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Gera gráfico comparativo de séries temporais (últimos 180 dias) e MAE por janela."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    recent = preds_df.tail(180)
    axes[0].plot(recent["data_referencia"], recent["y_true"], label="Observado (SE)", linewidth=1.8)
    axes[0].plot(
        recent["data_referencia"],
        recent["y_pred_baseline"],
        label="Baseline Semanal (Lag 7)",
        linestyle="--",
        alpha=0.8,
    )
    axes[0].plot(
        recent["data_referencia"],
        recent["y_pred_lgbm"],
        label="LightGBM (D+7)",
        linewidth=1.5,
    )
    axes[0].set_title("Carga Diária Subsistema SE — Últimos 180 Dias de Backtesting")
    axes[0].set_ylabel("Carga (MWmed)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    x = np.arange(len(folds_df))
    width = 0.38
    axes[1].bar(x - width / 2, folds_df["baseline_mae_mwmed"], width, label="Baseline Semanal")
    axes[1].bar(x + width / 2, folds_df["lgbm_mae_mwmed"], width, label="LightGBM (D+7)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [
            f"F{int(f)}\n{s[:7]}"
            for f, s in zip(folds_df["fold"], folds_df["test_start"], strict=False)
        ],
        fontsize=8,
    )
    axes[1].set_title("MAE por Janela Temporal de Backtesting (MWmed)")
    axes[1].set_ylabel("MAE (MWmed)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_walk_forward_backtest()
