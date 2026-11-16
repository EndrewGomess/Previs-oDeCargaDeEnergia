"""Testes unitários e de integração para o Baseline Ingênuo Semanal (Etapa 4)."""

import importlib

import pandas as pd

baseline_mod = importlib.import_module("src.models.04_baseline_semanal")
compute_weekly_baseline = baseline_mod.compute_weekly_baseline
evaluate_baseline_metrics = baseline_mod.evaluate_baseline_metrics
load_gold_se_data = baseline_mod.load_gold_se_data


def test_weekly_baseline_no_leakage_synthetic():
    """Garante que a previsão de t é exatamente o valor observado em t - 7 dias."""
    dates = pd.date_range("2025-01-01", periods=15, freq="D")
    loads = [float(1000 + i * 10) for i in range(15)]
    df = pd.DataFrame({"data_referencia": dates, "carga_energia_mwmed": loads})

    res = compute_weekly_baseline(df)

    # Primeiros 7 dias não possuem histórico de t-7
    assert res["y_pred_baseline"].iloc[:7].isna().all()
    # Dia índice 7 (2025-01-08) deve prever exatamente o valor do índice 0 (1000.0)
    assert res["y_pred_baseline"].iloc[7] == 1000.0
    # Dia índice 14 (2025-01-15) deve prever exatamente o valor do índice 7 (1070.0)
    assert res["y_pred_baseline"].iloc[14] == 1070.0


def test_weekly_baseline_on_gold_data():
    """Valida a execução do baseline sobre a tabela Gold real no DuckDB."""
    df_gold = load_gold_se_data()
    df_pred = compute_weekly_baseline(df_gold)
    metrics = evaluate_baseline_metrics(df_pred)

    assert metrics["n_obs"] == len(df_gold) - 7
    assert metrics["mae_mwmed"] > 0
    assert 0 < metrics["mape_pct"] < 20
