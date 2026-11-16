"""Testes unitários e de integração para o Backtesting Temporal (Etapa 6)."""

import importlib
from pathlib import Path

import pandas as pd

backtest_mod = importlib.import_module("src.evaluation.06_backtest")
generate_temporal_folds = backtest_mod.generate_temporal_folds
run_walk_forward_backtest = backtest_mod.run_walk_forward_backtest


def test_temporal_folds_strict_ordering():
    """Garante que todas as janelas de teste são estritamente sequenciais e futuras."""
    dates = pd.date_range("2020-01-01", "2026-11-14", freq="D")
    df_dummy = pd.DataFrame({"data_referencia": dates})

    folds = generate_temporal_folds(
        df_dummy, backtest_start="2024-01-01", fold_days=90, val_days=90
    )
    assert len(folds) >= 10
    for i, (val_start, test_start, test_end) in enumerate(folds):
        assert val_start < test_start <= test_end
        if i > 0:
            prev_end = folds[i - 1][2]
            assert test_start == prev_end + pd.Timedelta(days=1)


def test_walk_forward_backtest_end_to_end(tmp_path: Path):
    """Valida execução completa do backtesting walk-forward e geração de artefatos."""
    reports_dir = tmp_path / "reports"
    folds_df, preds_df, summary = run_walk_forward_backtest(
        reports_dir=reports_dir,
        backtest_start="2025-01-01",
        fold_days=180,
        val_days=90,
        log_mlflow=False,
    )
    assert len(folds_df) >= 3
    assert len(preds_df) > 500
    assert summary["lgbm_mae_mwmed"] > 0
    assert summary["baseline_mae_mwmed"] > 0
    assert (reports_dir / "06_backtest_folds.csv").exists()
    assert (reports_dir / "06_backtest_comparison.png").exists()
