#!/usr/bin/env python3
"""05_train_lightgbm.py

Etapa 5: Treinamento do modelo LightGBM para previsão de carga diária (horizonte D+7)
com split temporal rigoroso e registro completo de parâmetros, métricas,
importância de variáveis e artefato do modelo no MLflow.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

baseline_mod = importlib.import_module("src.models.04_baseline_semanal")
feat_mod = importlib.import_module("src.features.05_feature_engineering")

load_gold_se_data = baseline_mod.load_gold_se_data
build_features = feat_mod.build_features
FEATURE_COLUMNS = feat_mod.FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("models.lightgbm")

EXPERIMENT_NAME = "ons_carga_se_d7"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula MAE, MAPE (%) e RMSE."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae_mwmed": mae, "mape_pct": mape, "rmse_mwmed": rmse}


def prepare_modeling_dataset(warehouse_path: Path = Path("data/warehouse.duckdb")) -> pd.DataFrame:
    """Carrega Gold SE e gera features sem vazamento, descartando linhas de warmup."""
    df_gold = load_gold_se_data(warehouse_path=warehouse_path)
    df_feat = build_features(df_gold)
    df_model = df_feat.dropna(subset=FEATURE_COLUMNS + ["carga_energia_mwmed"]).reset_index(
        drop=True
    )
    return df_model


def train_and_log_lightgbm(
    warehouse_path: Path = Path("data/warehouse.duckdb"),
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    val_start_date: str = "2025-05-15",
    test_start_date: str = "2025-11-15",
    params: Dict[str, Any] | None = None,
) -> Tuple[lgb.LGBMRegressor, Dict[str, float]]:
    """Treina LightGBM em split temporal (treino / val early-stop / teste) e registra no MLflow."""
    df_model = prepare_modeling_dataset(warehouse_path=warehouse_path)

    val_dt = pd.Timestamp(val_start_date)
    test_dt = pd.Timestamp(test_start_date)

    train_df = df_model[df_model["data_referencia"] < val_dt].copy()
    val_df = df_model[
        (df_model["data_referencia"] >= val_dt) & (df_model["data_referencia"] < test_dt)
    ].copy()
    test_df = df_model[df_model["data_referencia"] >= test_dt].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Split temporal inválido: verifique datas val_start_date e test_start_date."
        )

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["carga_energia_mwmed"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["carga_energia_mwmed"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["carga_energia_mwmed"]

    lgbm_params = {**DEFAULT_LGBM_PARAMS, **(params or {})}

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="lightgbm_d7_temporal_holdout") as run:
        logger.info("Iniciando MLflow run_id=%s ...", run.info.run_id)
        logger.info(
            "Split temporal -> Treino: %d (%s a %s) | Val: %d | Teste: %d (%s a %s)",
            len(train_df),
            train_df["data_referencia"].min().date(),
            train_df["data_referencia"].max().date(),
            len(val_df),
            len(test_df),
            test_df["data_referencia"].min().date(),
            test_df["data_referencia"].max().date(),
        )

        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        # Predição no conjunto de teste temporal (futuro não visto)
        y_pred_lgbm = model.predict(X_test)
        y_pred_baseline = test_df["lag_7"].to_numpy()
        y_true = y_test.to_numpy()

        lgbm_metrics = compute_metrics(y_true, y_pred_lgbm)
        base_metrics = compute_metrics(y_true, y_pred_baseline)

        mae_gain_pct = (
            (base_metrics["mae_mwmed"] - lgbm_metrics["mae_mwmed"]) / base_metrics["mae_mwmed"]
        ) * 100.0
        mape_gain_pct = (
            (base_metrics["mape_pct"] - lgbm_metrics["mape_pct"]) / base_metrics["mape_pct"]
        ) * 100.0

        # Registro de parâmetros
        mlflow.log_params(
            {
                **lgbm_params,
                "n_features": len(FEATURE_COLUMNS),
                "best_iteration": model.best_iteration_,
                "val_start_date": val_start_date,
                "test_start_date": test_start_date,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
            }
        )

        # Registro de métricas
        mlflow.log_metrics(
            {
                "test_lgbm_mae_mwmed": lgbm_metrics["mae_mwmed"],
                "test_lgbm_mape_pct": lgbm_metrics["mape_pct"],
                "test_lgbm_rmse_mwmed": lgbm_metrics["rmse_mwmed"],
                "test_baseline_mae_mwmed": base_metrics["mae_mwmed"],
                "test_baseline_mape_pct": base_metrics["mape_pct"],
                "test_baseline_rmse_mwmed": base_metrics["rmse_mwmed"],
                "mae_improvement_pct": mae_gain_pct,
                "mape_improvement_pct": mape_gain_pct,
            }
        )

        # Artefato: importância das features
        fi_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance_gain": model.booster_.feature_importance(importance_type="gain"),
                "importance_split": model.booster_.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)
        fi_path = Path("data/feature_importance_etapa5.csv")
        fi_df.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))
        fi_path.unlink(missing_ok=True)

        # Registro do modelo no MLflow
        mlflow.lightgbm.log_model(
            model,
            name="lightgbm_model",
            input_example=X_test.head(5),
        )

        summary = {
            "run_id": run.info.run_id,
            "best_iteration": int(model.best_iteration_),
            "lgbm_mae": lgbm_metrics["mae_mwmed"],
            "lgbm_mape": lgbm_metrics["mape_pct"],
            "baseline_mae": base_metrics["mae_mwmed"],
            "baseline_mape": base_metrics["mape_pct"],
            "mae_improvement_pct": mae_gain_pct,
            "mape_improvement_pct": mape_gain_pct,
            "top_features": fi_df.head(5)["feature"].tolist(),
        }

        print("\n=== ETAPA 5: TREINAMENTO LIGHTGBM + MLFLOW (HOLDOUT TEMPORAL) ===")
        print(f"Janela de Teste: {test_start_date} até {test_df['data_referencia'].max().date()}")
        print(f"MLflow Run ID  : {summary['run_id']} | Best Iter: {summary['best_iteration']}")
        print(
            f"Baseline (SE)  : MAE = {summary['baseline_mae']:.2f} MWmed | "
            f"MAPE = {summary['baseline_mape']:.2f}%"
        )
        print(
            f"LightGBM (SE)  : MAE = {summary['lgbm_mae']:.2f} MWmed | "
            f"MAPE = {summary['lgbm_mape']:.2f}%"
        )
        print(
            f"Ganho s/ Base  : {summary['mae_improvement_pct']:+.2f}% em MAE | "
            f"{summary['mape_improvement_pct']:+.2f}% em MAPE"
        )
        print(f"Top 5 Features : {summary['top_features']}")
        print("=================================================================\n")

        return model, summary


if __name__ == "__main__":
    train_and_log_lightgbm()
