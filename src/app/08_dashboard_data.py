#!/usr/bin/env python3
"""08_dashboard_data.py

Camada de preparação de dados e inferência operacional (D+1 a D+7) para o Dashboard Streamlit.
Conecta-se diretamente ao warehouse DuckDB (camada Gold enriquecida com clima) e aos
artefatos reais de backtesting walk-forward gerados nas etapas 06 e 07.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, Tuple

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

feat_mod = importlib.import_module("src.features.05_feature_engineering")
train_mod = importlib.import_module("src.models.05_train_lightgbm")

FEATURE_COLUMNS = feat_mod.FEATURE_COLUMNS
build_features = feat_mod.build_features
DEFAULT_LGBM_PARAMS = train_mod.DEFAULT_LGBM_PARAMS

DEFAULT_WAREHOUSE = Path("data/warehouse.duckdb")
DEFAULT_REPORTS_DIR = Path("reports")


def load_historical_series(warehouse_path: Path = DEFAULT_WAREHOUSE) -> pd.DataFrame:
    """Carrega a série diária histórica e climática da camada Gold no DuckDB."""
    if not warehouse_path.exists():
        raise FileNotFoundError(
            f"Banco DuckDB não encontrado em {warehouse_path}. Execute o pipeline dbt primeiro."
        )

    con = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        df = con.execute("""
            SELECT
                data_referencia,
                id_subsistema,
                nom_subsistema,
                carga_energia_mwmed,
                regime_medicao_ons,
                temp_media_se,
                temp_max_se,
                temp_min_se
            FROM gold.gold_carga_energia_se_clima
            ORDER BY data_referencia ASC
        """).df()
    finally:
        con.close()

    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    return df.sort_values("data_referencia").reset_index(drop=True)


def load_backtest_artifacts(
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega as tabelas consolidadas de backtesting das etapas 06 e 07."""
    bt_preds_path = reports_dir / "06_backtest_predictions.csv"
    bt_folds_path = reports_dir / "06_backtest_folds.csv"
    clima_comp_path = reports_dir / "07_backtest_clima_comparison.csv"
    clima_folds_path = reports_dir / "07_backtest_clima_folds.csv"

    for path in [bt_preds_path, bt_folds_path, clima_comp_path, clima_folds_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Artefato de backtesting não encontrado: {path}. Execute as etapas 06 e 07."
            )

    bt_preds = pd.read_csv(bt_preds_path)
    bt_preds["data_referencia"] = pd.to_datetime(bt_preds["data_referencia"])

    bt_folds = pd.read_csv(bt_folds_path)
    clima_comp = pd.read_csv(clima_comp_path)
    clima_folds = pd.read_csv(clima_folds_path)

    return bt_preds, bt_folds, clima_comp, clima_folds


def generate_next_7_days_forecast(
    df_hist: pd.DataFrame,
    bt_preds: pd.DataFrame,
    horizon_days: int = 7,
) -> pd.DataFrame:
    """Gera previsão real para os próximos 7 dias (D+1 a D+7) sem vazamento temporal.

    Como todas as defasagens do modelo base são >= 7 dias, os atributos de D+1 até D+7
    dependem exclusivamente de observações já conhecidas até a última data histórica.
    O intervalo de confiança de 90% é estimado pelos quantis empíricos dos resíduos
    out-of-sample do backtesting walk-forward.
    """
    last_date = df_hist["data_referencia"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    df_future_stub = pd.DataFrame(
        {
            "data_referencia": future_dates,
            "id_subsistema": "SE",
            "nom_subsistema": "Sudeste/Centro-Oeste",
            "carga_energia_mwmed": np.nan,
            "regime_medicao_ons": "3_com_mmgd",
        }
    )

    df_combined = pd.concat(
        [
            df_hist[
                [
                    "data_referencia",
                    "id_subsistema",
                    "nom_subsistema",
                    "carga_energia_mwmed",
                    "regime_medicao_ons",
                ]
            ],
            df_future_stub,
        ],
        ignore_index=True,
    )

    df_feat = build_features(df_combined)

    # Treino histórico completo com validação temporal nos últimos 90 dias para early stopping
    df_train_all = df_feat[df_feat["data_referencia"] <= last_date].dropna(
        subset=FEATURE_COLUMNS + ["carga_energia_mwmed"]
    )
    val_cutoff = last_date - pd.Timedelta(days=90)
    train_part = df_train_all[df_train_all["data_referencia"] < val_cutoff]
    val_part = df_train_all[df_train_all["data_referencia"] >= val_cutoff]

    model = lgb.LGBMRegressor(**DEFAULT_LGBM_PARAMS)
    model.fit(
        train_part[FEATURE_COLUMNS],
        train_part["carga_energia_mwmed"],
        eval_X=val_part[FEATURE_COLUMNS],
        eval_y=val_part["carga_energia_mwmed"],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    df_horizon = df_feat[df_feat["data_referencia"] > last_date].copy()
    preds_lgbm = model.predict(df_horizon[FEATURE_COLUMNS])
    preds_baseline = df_horizon["lag_7"].to_numpy()

    # Quantis empíricos de erro (P5 e P95) sobre o histórico out-of-sample do backtesting
    residuals = (bt_preds["y_true"] - bt_preds["y_pred_lgbm"]).to_numpy()
    q_low = float(np.quantile(residuals, 0.05))
    q_high = float(np.quantile(residuals, 0.95))

    return pd.DataFrame(
        {
            "data_referencia": df_horizon["data_referencia"].to_numpy(),
            "y_pred_lgbm": preds_lgbm,
            "y_pred_baseline": preds_baseline,
            "ic_inferior_mwmed": preds_lgbm + q_low,
            "ic_superior_mwmed": preds_lgbm + q_high,
        }
    )


def build_unified_timeline(
    df_hist: pd.DataFrame,
    bt_preds: pd.DataFrame,
    df_forecast_7d: pd.DataFrame,
) -> pd.DataFrame:
    """Consolida histórico observado, previsões out-of-sample e projeção D+7 em uma tabela."""
    base = df_hist[
        [
            "data_referencia",
            "carga_energia_mwmed",
            "regime_medicao_ons",
            "temp_media_se",
            "temp_max_se",
        ]
    ].rename(columns={"carga_energia_mwmed": "carga_real_mwmed"})

    bt_cols = bt_preds[["data_referencia", "fold", "y_pred_baseline", "y_pred_lgbm"]].rename(
        columns={
            "y_pred_baseline": "baseline_semanal_mwmed",
            "y_pred_lgbm": "carga_prevista_lgbm_mwmed",
        }
    )

    timeline = base.merge(bt_cols, on="data_referencia", how="left")
    timeline["ic_inferior_mwmed"] = np.nan
    timeline["ic_superior_mwmed"] = np.nan
    timeline["tipo_registro"] = np.where(
        timeline["carga_prevista_lgbm_mwmed"].notna(),
        "Backtesting Out-of-Sample",
        "Historico Observado",
    )

    future_rows = pd.DataFrame(
        {
            "data_referencia": df_forecast_7d["data_referencia"],
            "carga_real_mwmed": np.nan,
            "regime_medicao_ons": "3_com_mmgd",
            "temp_media_se": np.nan,
            "temp_max_se": np.nan,
            "fold": np.nan,
            "baseline_semanal_mwmed": df_forecast_7d["y_pred_baseline"],
            "carga_prevista_lgbm_mwmed": df_forecast_7d["y_pred_lgbm"],
            "ic_inferior_mwmed": df_forecast_7d["ic_inferior_mwmed"],
            "ic_superior_mwmed": df_forecast_7d["ic_superior_mwmed"],
            "tipo_registro": "Previsao D+7",
        }
    )

    unified = pd.concat([timeline, future_rows], ignore_index=True)
    unified = unified.sort_values("data_referencia").reset_index(drop=True)

    # Métricas de erro para registros com valor real e previsão disponíveis
    unified["residuo_lgbm"] = unified["carga_real_mwmed"] - unified["carga_prevista_lgbm_mwmed"]
    unified["erro_absoluto_lgbm"] = unified["residuo_lgbm"].abs()
    unified["erro_percentual_lgbm"] = (
        unified["erro_absoluto_lgbm"] / unified["carga_real_mwmed"]
    ) * 100.0
    unified["erro_absoluto_baseline"] = (
        unified["carga_real_mwmed"] - unified["baseline_semanal_mwmed"]
    ).abs()
    unified["erro_percentual_baseline"] = (
        unified["erro_absoluto_baseline"] / unified["carga_real_mwmed"]
    ) * 100.0

    return unified


def compute_kpis(df_filtered: pd.DataFrame, df_forecast_7d: pd.DataFrame) -> Dict[str, object]:
    """Calcula os KPIs dinâmicos do período selecionado e o pico previsto para D+1..D+7."""
    eval_df = df_filtered.dropna(subset=["carga_real_mwmed", "carga_prevista_lgbm_mwmed"])

    if eval_df.empty:
        lgbm_mae = lgbm_mape = lgbm_rmse = 0.0
        base_mae = base_mape = base_rmse = 0.0
        mae_gain_pct = mape_gain_pct = mae_diff_mwmed = 0.0
        n_eval = 0
    else:
        y_true = eval_df["carga_real_mwmed"].to_numpy()
        y_lgbm = eval_df["carga_prevista_lgbm_mwmed"].to_numpy()
        y_base = eval_df["baseline_semanal_mwmed"].to_numpy()

        lgbm_mae = float(np.mean(np.abs(y_true - y_lgbm)))
        lgbm_mape = float(np.mean(np.abs((y_true - y_lgbm) / y_true)) * 100.0)
        lgbm_rmse = float(np.sqrt(np.mean((y_true - y_lgbm) ** 2)))

        base_mae = float(np.mean(np.abs(y_true - y_base)))
        base_mape = float(np.mean(np.abs((y_true - y_base) / y_true)) * 100.0)
        base_rmse = float(np.sqrt(np.mean((y_true - y_base) ** 2)))

        mae_diff_mwmed = base_mae - lgbm_mae
        mae_gain_pct = (mae_diff_mwmed / base_mae) * 100.0 if base_mae > 0 else 0.0
        mape_gain_pct = ((base_mape - lgbm_mape) / base_mape) * 100.0 if base_mape > 0 else 0.0
        n_eval = int(len(eval_df))

    peak_idx = df_forecast_7d["y_pred_lgbm"].idxmax()
    peak_val = float(df_forecast_7d.loc[peak_idx, "y_pred_lgbm"])
    peak_date = str(pd.Timestamp(df_forecast_7d.loc[peak_idx, "data_referencia"]).date())

    return {
        "n_eval": n_eval,
        "lgbm_mae": lgbm_mae,
        "lgbm_mape": lgbm_mape,
        "lgbm_rmse": lgbm_rmse,
        "base_mae": base_mae,
        "base_mape": base_mape,
        "base_rmse": base_rmse,
        "mae_diff_mwmed": mae_diff_mwmed,
        "mae_gain_pct": mae_gain_pct,
        "mape_gain_pct": mape_gain_pct,
        "peak_next_7d_mwmed": peak_val,
        "peak_next_7d_date": peak_date,
    }
