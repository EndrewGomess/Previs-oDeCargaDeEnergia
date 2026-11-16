"""Testes unitários e de integração para Engenharia de Atributos e LightGBM + MLflow (Etapa 5)."""

import importlib
from pathlib import Path

import mlflow
import pandas as pd

baseline_mod = importlib.import_module("src.models.04_baseline_semanal")
feat_mod = importlib.import_module("src.features.05_feature_engineering")
train_mod = importlib.import_module("src.models.05_train_lightgbm")

load_gold_se_data = baseline_mod.load_gold_se_data
build_features = feat_mod.build_features
FEATURE_COLUMNS = feat_mod.FEATURE_COLUMNS
train_and_log_lightgbm = train_mod.train_and_log_lightgbm


def test_no_data_leakage_from_recent_days():
    """Prova matemática contra vazamento: alterar os dias t-6 até t não muda as features de t."""
    df_gold = load_gold_se_data().tail(450).reset_index(drop=True)
    target_idx = len(df_gold) - 1

    feat_original = build_features(df_gold)

    # Perturba os últimos 7 dias (t-6 até t) com valores extremos
    df_perturbed = df_gold.copy()
    df_perturbed.loc[target_idx - 6 : target_idx, "carga_energia_mwmed"] = 999999.0
    feat_perturbed = build_features(df_perturbed)

    row_orig = feat_original.iloc[target_idx][FEATURE_COLUMNS]
    row_pert = feat_perturbed.iloc[target_idx][FEATURE_COLUMNS]

    pd.testing.assert_series_equal(row_orig, row_pert, check_exact=True)


def test_features_no_nulls_after_warmup():
    """Garante ausência de nulos nas features após o período de aquecimento de 364 dias."""
    df_gold = load_gold_se_data()
    df_feat = build_features(df_gold)
    after_warmup = df_feat.iloc[364:]
    assert after_warmup[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_lightgbm_training_and_mlflow_logging(tmp_path: Path):
    """Testa treinamento do LightGBM e registro completo em banco MLflow isolado."""
    db_uri = f"sqlite:///{tmp_path / 'test_mlflow.db'}"
    model, summary = train_and_log_lightgbm(
        tracking_uri=db_uri,
        experiment_name="test_ons_carga",
    )
    assert model is not None
    assert summary["lgbm_mae"] > 0
    assert summary["best_iteration"] > 0

    mlflow.set_tracking_uri(db_uri)
    run = mlflow.get_run(summary["run_id"])
    assert run.info.status == "FINISHED"
    assert "test_lgbm_mae_mwmed" in run.data.metrics
