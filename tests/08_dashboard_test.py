"""Testes unitários e de integração para a camada de Dashboard Streamlit (Etapa 8)."""

import importlib
from pathlib import Path

import pandas as pd

dash_data = importlib.import_module("src.app.08_dashboard_data")
dash_app = importlib.import_module("src.app.08_dashboard_app")


def test_load_historical_and_backtest_artifacts():
    """Valida leitura da série Gold climática e dos CSVs de backtesting."""
    df_hist = dash_data.load_historical_series()
    bt_preds, bt_folds, clima_comp, clima_folds = dash_data.load_backtest_artifacts()

    assert len(df_hist) >= 3240
    assert {"data_referencia", "carga_energia_mwmed", "temp_media_se"}.issubset(df_hist.columns)
    assert len(bt_folds) == 12
    assert len(bt_preds) > 1000
    assert len(clima_comp) == 5
    assert len(clima_folds) == 12


def test_generate_next_7_days_forecast_leakage_free():
    """Garante previsão D+1..D+7 sem nulos, consistente e com intervalo de confiança válido."""
    df_hist = dash_data.load_historical_series()
    bt_preds, _, _, _ = dash_data.load_backtest_artifacts()

    df_7d = dash_data.generate_next_7_days_forecast(df_hist, bt_preds, horizon_days=7)

    assert len(df_7d) == 7
    assert df_7d.isna().sum().sum() == 0
    last_hist_date = df_hist["data_referencia"].max()
    assert df_7d["data_referencia"].min() == last_hist_date + pd.Timedelta(days=1)
    assert df_7d["data_referencia"].max() == last_hist_date + pd.Timedelta(days=7)
    assert (df_7d["ic_inferior_mwmed"] < df_7d["y_pred_lgbm"]).all()
    assert (df_7d["y_pred_lgbm"] < df_7d["ic_superior_mwmed"]).all()


def test_unified_timeline_kpis_and_figures():
    """Valida consolidação da linha do tempo, KPIs dinâmicos e criação das figuras Plotly."""
    df_hist = dash_data.load_historical_series()
    bt_preds, bt_folds, clima_comp, _ = dash_data.load_backtest_artifacts()
    df_7d = dash_data.generate_next_7_days_forecast(df_hist, bt_preds, horizon_days=7)

    timeline = dash_data.build_unified_timeline(df_hist, bt_preds, df_7d)
    assert len(timeline) == len(df_hist) + 7
    assert (timeline["tipo_registro"] == "Previsao D+7").sum() == 7

    kpis = dash_data.compute_kpis(timeline, df_7d)
    assert kpis["n_eval"] == len(bt_preds)
    assert kpis["lgbm_mae"] < kpis["base_mae"]
    assert kpis["lgbm_mape"] < kpis["base_mape"]
    assert kpis["peak_next_7d_mwmed"] == df_7d["y_pred_lgbm"].max()

    eval_df = timeline.dropna(subset=["carga_real_mwmed", "carga_prevista_lgbm_mwmed"])
    fig_main = dash_app.build_main_timeline_figure(timeline.tail(120), True, True)
    fig_err = dash_app.build_error_timeseries_figure(eval_df.tail(90))
    fig_res = dash_app.build_residual_histogram_figure(eval_df)
    fig_folds = dash_app.build_folds_bar_figure(bt_folds)
    fig_clima_corr = dash_app.build_climate_correlation_figure(timeline.tail(120))
    fig_clima_scen = dash_app.build_climate_scenarios_figure(clima_comp)

    assert len(fig_main.data) >= 4
    assert len(fig_err.data) == 2
    assert len(fig_res.data) == 2
    assert len(fig_folds.data) == 2
    assert len(fig_clima_corr.data) == 2
    assert len(fig_clima_scen.data) == 1


def test_streamlit_apptest_smoke():
    """Executa app.py via Streamlit AppTest e valida renderização sem exceções."""
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    assert not at.exception
    assert not at.error
    assert len(at.metric) == 4
    assert len(at.tabs) == 3


def test_dashboard_launcher_helpers():
    """Valida utilitários de porta livre, verificação do DuckDB e argumentos do launcher."""
    launcher = importlib.import_module("launch_dashboard")

    port = launcher.find_free_port(8501)
    assert 8501 <= port < 8521

    venv_py = launcher.get_venv_python()
    assert launcher.is_gold_warehouse_ready(venv_py)

    parsed = launcher.parse_args(["--port", "8505", "--no-browser", "--force-refresh"])
    assert parsed.port == 8505
    assert parsed.no_browser is True
    assert parsed.force_refresh is True
