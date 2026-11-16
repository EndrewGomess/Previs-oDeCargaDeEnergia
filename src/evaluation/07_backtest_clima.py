#!/usr/bin/env python3
"""07_backtest_clima.py

Bônus: Backtesting temporal comparando o impacto da variável exógena Temperatura
no modelo LightGBM ao longo das mesmas 12 janelas (2024-01-01 a 2026-11-14).

Cenários avaliados honestamente:
1. Baseline Ingênuo Semanal (lag 7)
2. LightGBM Sem Clima (referência da Etapa 6)
3. LightGBM + Clima (Apenas Lags >= 7 - zero previsão futura)
4. LightGBM + Clima (Previsão D+7 Simulada com erro realista sigma=1.5°C no teste)
5. LightGBM + Clima (Temperatura Realizada ERA5 - teto teórico com clima perfeito)
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

train_mod = importlib.import_module("src.models.05_train_lightgbm")
bt_mod = importlib.import_module("src.evaluation.06_backtest")
weather_feat_mod = importlib.import_module("src.features.07_feature_engineering_clima")

DEFAULT_LGBM_PARAMS = train_mod.DEFAULT_LGBM_PARAMS
DEFAULT_TRACKING_URI = train_mod.DEFAULT_TRACKING_URI
EXPERIMENT_NAME = train_mod.EXPERIMENT_NAME
compute_metrics = train_mod.compute_metrics
generate_temporal_folds = bt_mod.generate_temporal_folds

BASE_FEATURES = weather_feat_mod.BASE_FEATURES
WEATHER_LAG_FEATURES = weather_feat_mod.WEATHER_LAG_FEATURES
WEATHER_FORECAST_FEATURES = weather_feat_mod.WEATHER_FORECAST_FEATURES
load_gold_se_with_weather = weather_feat_mod.load_gold_se_with_weather
build_weather_features = weather_feat_mod.build_weather_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluation.backtest_clima")


def _evaluate_scenario(
    df_train_source: pd.DataFrame,
    df_test_source: pd.DataFrame,
    feature_cols: List[str],
    folds: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Executa walk-forward para um conjunto de features e fontes de treino/teste."""
    fold_records: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []

    for idx, (val_start, test_start, test_end) in enumerate(folds, start=1):
        train_df = df_train_source[df_train_source["data_referencia"] < val_start]
        val_df = df_train_source[
            (df_train_source["data_referencia"] >= val_start)
            & (df_train_source["data_referencia"] < test_start)
        ]
        test_df = df_test_source[
            (df_test_source["data_referencia"] >= test_start)
            & (df_test_source["data_referencia"] <= test_end)
        ]

        model = lgb.LGBMRegressor(**DEFAULT_LGBM_PARAMS)
        model.fit(
            train_df[feature_cols],
            train_df["carga_energia_mwmed"],
            eval_X=val_df[feature_cols],
            eval_y=val_df["carga_energia_mwmed"],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        y_true = test_df["carga_energia_mwmed"].to_numpy()
        y_pred = model.predict(test_df[feature_cols])
        y_base = test_df["lag_7"].to_numpy()

        m_mod = compute_metrics(y_true, y_pred)
        m_base = compute_metrics(y_true, y_base)
        fold_records.append(
            {
                "fold": idx,
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "baseline_mae": m_base["mae_mwmed"],
                "model_mae": m_mod["mae_mwmed"],
                "model_mape": m_mod["mape_pct"],
            }
        )
        pred_frames.append(
            pd.DataFrame(
                {
                    "fold": idx,
                    "data_referencia": test_df["data_referencia"].to_numpy(),
                    "y_true": y_true,
                    "y_pred_baseline": y_base,
                    "y_pred": y_pred,
                }
            )
        )

    folds_df = pd.DataFrame(fold_records)
    preds_df = pd.concat(pred_frames, ignore_index=True)
    overall = compute_metrics(preds_df["y_true"].to_numpy(), preds_df["y_pred"].to_numpy())
    base_overall = compute_metrics(
        preds_df["y_true"].to_numpy(), preds_df["y_pred_baseline"].to_numpy()
    )

    overall["baseline_mae_mwmed"] = base_overall["mae_mwmed"]
    overall["baseline_mape_pct"] = base_overall["mape_pct"]
    overall["mae_gain_vs_base_pct"] = (
        (base_overall["mae_mwmed"] - overall["mae_mwmed"]) / base_overall["mae_mwmed"]
    ) * 100.0
    overall["mape_gain_vs_base_pct"] = (
        (base_overall["mape_pct"] - overall["mape_pct"]) / base_overall["mape_pct"]
    ) * 100.0
    overall["folds_won"] = int((folds_df["model_mae"] < folds_df["baseline_mae"]).sum())
    overall["n_folds"] = int(len(folds_df))
    return folds_df, preds_df, overall


def run_weather_backtest(
    warehouse_path: Path = Path("data/warehouse.duckdb"),
    reports_dir: Path = Path("reports"),
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    backtest_start: str = "2024-01-01",
    fold_days: int = 90,
    val_days: int = 90,
    log_mlflow: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executa comparação completa com e sem variável exógena de temperatura."""
    df_gold_clima = load_gold_se_with_weather(warehouse_path=warehouse_path)

    all_weather_cols = BASE_FEATURES + WEATHER_LAG_FEATURES + WEATHER_FORECAST_FEATURES
    df_clean = (
        build_weather_features(df_gold_clima, forecast_noise_std=0.0)
        .dropna(subset=all_weather_cols + ["carga_energia_mwmed"])
        .reset_index(drop=True)
    )
    df_noisy = (
        build_weather_features(df_gold_clima, forecast_noise_std=1.5, noise_seed=42)
        .dropna(subset=all_weather_cols + ["carga_energia_mwmed"])
        .reset_index(drop=True)
    )

    folds = generate_temporal_folds(
        df_clean,
        backtest_start=backtest_start,
        fold_days=fold_days,
        val_days=val_days,
    )

    scenarios = [
        ("LightGBM Sem Clima (Ref.)", df_clean, df_clean, BASE_FEATURES),
        (
            "LightGBM + Clima (Lags >= 7)",
            df_clean,
            df_clean,
            BASE_FEATURES + WEATHER_LAG_FEATURES,
        ),
        (
            "LightGBM + Clima (Prev. D+7 σ=1.5°C)",
            df_clean,
            df_noisy,
            all_weather_cols,
        ),
        (
            "LightGBM + Clima (ERA5 Realizado)",
            df_clean,
            df_clean,
            all_weather_cols,
        ),
    ]

    summary_rows: List[Dict[str, object]] = []
    fold_tables: Dict[str, pd.DataFrame] = {}

    for label, tr_df, te_df, feat_cols in scenarios:
        logger.info("Avaliando cenário: %s (%d features) ...", label, len(feat_cols))
        f_df, _, metrics = _evaluate_scenario(tr_df, te_df, feat_cols, folds)
        fold_tables[label] = f_df
        summary_rows.append(
            {
                "modelo": label,
                "n_features": len(feat_cols),
                "mae_mwmed": metrics["mae_mwmed"],
                "mape_pct": metrics["mape_pct"],
                "rmse_mwmed": metrics["rmse_mwmed"],
                "ganho_mae_vs_base_pct": metrics["mae_gain_vs_base_pct"],
                "ganho_mape_vs_base_pct": metrics["mape_gain_vs_base_pct"],
                "folds_vencidos": f"{metrics['folds_won']}/{metrics['n_folds']}",
            }
        )

    base_row = {
        "modelo": "Baseline Ingênuo (Semanal)",
        "n_features": 1,
        "mae_mwmed": fold_tables["LightGBM Sem Clima (Ref.)"]["baseline_mae"].mean(),
        "mape_pct": 0.0,
        "rmse_mwmed": 0.0,
        "ganho_mae_vs_base_pct": 0.0,
        "ganho_mape_vs_base_pct": 0.0,
        "folds_vencidos": "-",
    }
    # Métricas exatas do baseline sobre todos os dias
    base_df_preds = pd.concat(
        [
            df_clean[(df_clean["data_referencia"] >= ts) & (df_clean["data_referencia"] <= te)][
                ["carga_energia_mwmed", "lag_7"]
            ]
            for _, ts, te in folds
        ],
        ignore_index=True,
    )
    base_exact = compute_metrics(
        base_df_preds["carga_energia_mwmed"].to_numpy(),
        base_df_preds["lag_7"].to_numpy(),
    )
    base_row["mae_mwmed"] = base_exact["mae_mwmed"]
    base_row["mape_pct"] = base_exact["mape_pct"]
    base_row["rmse_mwmed"] = base_exact["rmse_mwmed"]

    comparison_df = pd.DataFrame([base_row, *summary_rows])

    # Combina MAE por fold para comparar especificamente o Fold 5 (verão 2025)
    merged_folds = fold_tables["LightGBM Sem Clima (Ref.)"][
        ["fold", "test_start", "test_end", "baseline_mae", "model_mae"]
    ].rename(columns={"model_mae": "lgbm_sem_clima_mae"})
    merged_folds["lgbm_clima_lag7_mae"] = fold_tables["LightGBM + Clima (Lags >= 7)"]["model_mae"]
    merged_folds["lgbm_clima_prev_15c_mae"] = fold_tables["LightGBM + Clima (Prev. D+7 σ=1.5°C)"][
        "model_mae"
    ]
    merged_folds["lgbm_clima_era5_mae"] = fold_tables["LightGBM + Clima (ERA5 Realizado)"][
        "model_mae"
    ]

    reports_dir.mkdir(parents=True, exist_ok=True)
    comp_csv = reports_dir / "07_backtest_clima_comparison.csv"
    folds_csv = reports_dir / "07_backtest_clima_folds.csv"
    chart_png = reports_dir / "07_backtest_clima_chart.png"

    comparison_df.to_csv(comp_csv, index=False)
    merged_folds.to_csv(folds_csv, index=False)
    _save_weather_chart(merged_folds, comparison_df, chart_png)

    if log_mlflow:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        for row in summary_rows:
            run_name = f"backtest_clima_{row['modelo'].split('(')[1].replace(')', '').strip()}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({"modelo": row["modelo"], "n_features": row["n_features"]})
                mlflow.log_metrics(
                    {
                        "bt_mae_mwmed": float(row["mae_mwmed"]),
                        "bt_mape_pct": float(row["mape_pct"]),
                        "bt_rmse_mwmed": float(row["rmse_mwmed"]),
                        "bt_ganho_mae_vs_base_pct": float(row["ganho_mae_vs_base_pct"]),
                    }
                )
                mlflow.log_artifact(str(comp_csv))
                mlflow.log_artifact(str(chart_png))

    print("\n=== BÔNUS: IMPACTO DA TEMPERATURA EXÓGENA NO BACKTESTING (12 JANELAS) ===")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n--- MAE por Janela (Destaque para Fold 5: Verão 2025) ---")
    print(merged_folds.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("=========================================================================\n")

    return comparison_df, merged_folds


def _save_weather_chart(
    merged_folds: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Salva gráfico comparando MAE por fold e consolidado com e sem temperatura."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(merged_folds))
    w = 0.25
    axes[0].bar(x - w, merged_folds["baseline_mae"], w, label="Baseline Semanal")
    axes[0].bar(x, merged_folds["lgbm_sem_clima_mae"], w, label="LightGBM Sem Clima")
    axes[0].bar(
        x + w,
        merged_folds["lgbm_clima_prev_15c_mae"],
        w,
        label="LightGBM + Clima (Prev. σ=1.5°C)",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"F{int(f)}" for f in merged_folds["fold"]])
    axes[0].set_title("MAE por Janela (Impacto do Clima no Verão/Fold 5)")
    axes[0].set_ylabel("MAE (MWmed)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    colors = ["#7f8c8d", "#3498db", "#1abc9c", "#2ecc71", "#27ae60"]
    axes[1].barh(comparison_df["modelo"], comparison_df["mae_mwmed"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_title("MAE Consolidado Out-of-Sample (1.049 dias)")
    axes[1].set_xlabel("MAE (MWmed)")
    for i, val in enumerate(comparison_df["mae_mwmed"]):
        axes[1].text(val + 15, i, f"{val:.1f}", va="center", fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_weather_backtest()
