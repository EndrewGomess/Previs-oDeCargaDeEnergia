"""Testes unitários e de integração para o Bônus de Temperatura Exógena (07_)."""

import importlib
from pathlib import Path

import duckdb
import pandas as pd

weather_feat_mod = importlib.import_module("src.features.07_feature_engineering_clima")
weather_bt_mod = importlib.import_module("src.evaluation.07_backtest_clima")

WEATHER_LAG_FEATURES = weather_feat_mod.WEATHER_LAG_FEATURES
load_gold_se_with_weather = weather_feat_mod.load_gold_se_with_weather
build_weather_features = weather_feat_mod.build_weather_features
run_weather_backtest = weather_bt_mod.run_weather_backtest


def test_weather_bronze_and_gold_integrity():
    """Garante integridade do Parquet climático Bronze e da tabela Gold enriquecida."""
    bronze_weather = Path("data/bronze/TEMPERATURA_SE_2018_2026.parquet")
    assert bronze_weather.exists() and bronze_weather.stat().st_size > 0

    con = duckdb.connect("data/warehouse.duckdb", read_only=True)
    try:
        res = con.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN temp_media_se IS NULL THEN 1 ELSE 0 END) AS null_temp
            FROM gold.gold_carga_energia_se_clima
        """).fetchone()
        assert res[0] >= 3240
        assert res[1] == 0
    finally:
        con.close()


def test_weather_lag_features_no_leakage():
    """Prova que perturbar temperaturas de t-6 até t não altera WEATHER_LAG_FEATURES em t."""
    df = load_gold_se_with_weather().tail(450).reset_index(drop=True)
    target_idx = len(df) - 1

    orig = build_weather_features(df)
    df_pert = df.copy()
    df_pert.loc[target_idx - 6 : target_idx, ["temp_media_se", "temp_max_se", "temp_max_sp"]] = 50.0
    pert = build_weather_features(df_pert)

    pd.testing.assert_series_equal(
        orig.iloc[target_idx][WEATHER_LAG_FEATURES],
        pert.iloc[target_idx][WEATHER_LAG_FEATURES],
        check_exact=True,
    )


def test_weather_backtest_runs_cleanly(tmp_path: Path):
    """Valida execução reduzida do backtest climático sem erros."""
    comp_df, folds_df = run_weather_backtest(
        reports_dir=tmp_path / "reports",
        backtest_start="2025-06-01",
        fold_days=180,
        val_days=90,
        log_mlflow=False,
    )
    assert len(comp_df) == 5
    assert len(folds_df) >= 2
    assert (tmp_path / "reports" / "07_backtest_clima_chart.png").exists()
