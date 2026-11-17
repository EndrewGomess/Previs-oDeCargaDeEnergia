#!/usr/bin/env python3
"""08_dashboard_app.py

Interface interativa em Streamlit e Plotly para apresentação dos resultados do modelo
de Previsão de Carga Diária de Energia (horizonte D+7) do Subsistema Sudeste/Centro-Oeste (ONS).
"""

from __future__ import annotations

import importlib
from datetime import date
from typing import Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

data_mod = importlib.import_module("src.app.08_dashboard_data")
load_historical_series = data_mod.load_historical_series
load_backtest_artifacts = data_mod.load_backtest_artifacts
generate_next_7_days_forecast = data_mod.generate_next_7_days_forecast
build_unified_timeline = data_mod.build_unified_timeline
compute_kpis = data_mod.compute_kpis

# Paleta corporativa definida para contraste e clareza visual
COLOR_REAL = "#1f77b4"
COLOR_FORECAST = "#ff7f0e"
COLOR_CI_FILL = "rgba(255, 127, 14, 0.16)"
COLOR_BASELINE = "#6c757d"
COLOR_ERROR = "#d62728"
COLOR_TEMP = "#2ca02c"
COLOR_GRID = "#eef1f5"

REGIME_LABELS: Dict[str, str] = {
    "1_despachada": "Regime 1: Usinas Despachadas (até fev/2021)",
    "2_nao_despachada": "Regime 2: Inclui Não Despachadas (mar/2021 a abr/2023)",
    "3_com_mmgd": "Regime 3: Inclui Estimativa MMGD (a partir de abr/2023)",
}

REGIME_TRANSITIONS = [
    ("2021-03-01", "Transição ONS: Não Despachadas"),
    ("2023-04-29", "Transição ONS: Inclusão MMGD"),
]


@st.cache_data(show_spinner="Carregando dados históricos e gerando previsão D+7...")
def load_dashboard_bundle() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Carrega e armazena em cache os dados do DuckDB, artefatos de backtesting e previsão D+7."""
    df_hist = load_historical_series()
    bt_preds, bt_folds, clima_comp, clima_folds = load_backtest_artifacts()
    df_forecast_7d = generate_next_7_days_forecast(df_hist, bt_preds, horizon_days=7)
    df_timeline = build_unified_timeline(df_hist, bt_preds, df_forecast_7d)
    return df_timeline, df_forecast_7d, bt_folds, clima_comp, clima_folds, df_hist


def _apply_clean_style() -> None:
    """Aplica ajustes sutis de CSS para manter o layout corporativo limpo e bem espaçado."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.75rem;
                padding-bottom: 2.5rem;
                max-width: 1400px;
            }
            div[data-testid="stMetric"] {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 0.9rem 1rem;
            }
            div[data-testid="stMetricLabel"] {
                font-weight: 600;
                color: #334155;
            }
            .dashboard-subtitle {
                color: #475569;
                font-size: 1.02rem;
                margin-top: -0.5rem;
                margin-bottom: 1.25rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_pt_number(value: float, decimals: int = 1) -> str:
    """Formata números com separador de milhar em ponto e decimal em vírgula."""
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def build_main_timeline_figure(
    df_filtered: pd.DataFrame,
    show_baseline: bool,
    show_ci: bool,
) -> go.Figure:
    """Constrói o gráfico principal de série histórica, backtesting out-of-sample e previsão D+7."""
    fig = go.Figure()

    # Faixa de intervalo de confiança de 90% na projeção D+7
    ci_df = df_filtered.dropna(subset=["ic_inferior_mwmed", "ic_superior_mwmed"])
    if show_ci and not ci_df.empty:
        fig.add_trace(
            go.Scatter(
                x=ci_df["data_referencia"],
                y=ci_df["ic_superior_mwmed"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="IC 90% Superior",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ci_df["data_referencia"],
                y=ci_df["ic_inferior_mwmed"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=COLOR_CI_FILL,
                name="Intervalo de Confiança (90%)",
                hovertemplate=(
                    "Data: %{x|%d/%m/%Y}<br>Limite Inferior (90%): %{y:,.1f} MWmed<extra></extra>"
                ),
            )
        )

    # Série observada pelo ONS
    real_df = df_filtered.dropna(subset=["carga_real_mwmed"])
    if not real_df.empty:
        fig.add_trace(
            go.Scatter(
                x=real_df["data_referencia"],
                y=real_df["carga_real_mwmed"],
                mode="lines",
                name="Carga Real (ONS)",
                line=dict(color=COLOR_REAL, width=2.2),
                hovertemplate=("Data: %{x|%d/%m/%Y}<br>Carga Real: %{y:,.1f} MWmed<extra></extra>"),
            )
        )

    # Previsão contínua LightGBM (backtesting out-of-sample + horizonte D+1..D+7)
    pred_df = df_filtered.dropna(subset=["carga_prevista_lgbm_mwmed"])
    if not pred_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pred_df["data_referencia"],
                y=pred_df["carga_prevista_lgbm_mwmed"],
                mode="lines",
                name="Carga Prevista (LightGBM D+7)",
                line=dict(color=COLOR_FORECAST, width=2.2, dash="dash"),
                hovertemplate=(
                    "Data: %{x|%d/%m/%Y}<br>Carga Prevista (LightGBM): %{y:,.1f} MWmed"
                    "<extra></extra>"
                ),
            )
        )

    # Referência do Baseline Ingênuo Semanal
    base_df = df_filtered.dropna(subset=["baseline_semanal_mwmed"])
    if show_baseline and not base_df.empty:
        fig.add_trace(
            go.Scatter(
                x=base_df["data_referencia"],
                y=base_df["baseline_semanal_mwmed"],
                mode="lines",
                name="Baseline Ingênuo Semanal (t-7)",
                line=dict(color=COLOR_BASELINE, width=1.6, dash="dot"),
                opacity=0.85,
                hovertemplate=(
                    "Data: %{x|%d/%m/%Y}<br>Baseline Semanal: %{y:,.1f} MWmed<extra></extra>"
                ),
            )
        )

    # Marcadores verticais das mudanças metodológicas do ONS se estiverem no intervalo
    if not df_filtered.empty:
        min_dt = df_filtered["data_referencia"].min()
        max_dt = df_filtered["data_referencia"].max()

        for dt_str, label in REGIME_TRANSITIONS:
            dt_val = pd.Timestamp(dt_str)
            if min_dt <= dt_val <= max_dt:
                fig.add_vline(
                    x=dt_val.timestamp() * 1000,
                    line_width=1,
                    line_dash="dot",
                    line_color="#94a3b8",
                    annotation_text=label,
                    annotation_position="top left",
                    annotation_font_size=11,
                    annotation_font_color="#475569",
                )

        # Marcador vertical indicando o início do horizonte futuro D+1..D+7
        future_rows = df_filtered[df_filtered["tipo_registro"] == "Previsao D+7"]
        if not future_rows.empty:
            horizon_start = future_rows["data_referencia"].min()
            fig.add_vline(
                x=horizon_start.timestamp() * 1000,
                line_width=1.4,
                line_dash="dash",
                line_color=COLOR_FORECAST,
                annotation_text="Início Previsão D+7",
                annotation_position="top right",
                annotation_font_size=11,
                annotation_font_color=COLOR_FORECAST,
            )

    fig.update_layout(
        template="plotly_white",
        height=470,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        xaxis=dict(
            title="Data de Referência",
            showgrid=False,
            showline=True,
            linecolor="#cbd5e1",
        ),
        yaxis=dict(
            title="Carga de Energia (MWmed)",
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            tickformat=",.0f",
        ),
    )
    return fig


def build_error_timeseries_figure(df_eval: pd.DataFrame) -> go.Figure:
    """Constrói gráfico de barras/área do erro absoluto diário destacando picos de erro."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_eval["data_referencia"],
            y=df_eval["erro_absoluto_lgbm"],
            mode="lines",
            name="Erro Absoluto LightGBM",
            line=dict(color=COLOR_ERROR, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.14)",
            customdata=df_eval["erro_percentual_lgbm"],
            hovertemplate=(
                "Data: %{x|%d/%m/%Y}<br>Erro Absoluto: %{y:,.1f} MWmed"
                "<br>Erro Percentual: %{customdata:.2f}%<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_eval["data_referencia"],
            y=df_eval["erro_absoluto_baseline"],
            mode="lines",
            name="Erro Absoluto Baseline",
            line=dict(color=COLOR_BASELINE, width=1.4, dash="dot"),
            opacity=0.8,
            hovertemplate=("Data: %{x|%d/%m/%Y}<br>Erro Baseline: %{y:,.1f} MWmed<extra></extra>"),
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="Evolução Diária do Erro Absoluto (MWmed)",
        height=380,
        margin=dict(l=20, r=20, t=45, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Data", showgrid=False, linecolor="#cbd5e1"),
        yaxis=dict(
            title="Erro Absoluto (MWmed)",
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat=",.0f",
        ),
    )
    return fig


def build_residual_histogram_figure(df_eval: pd.DataFrame) -> go.Figure:
    """Constrói histograma comparativo da distribuição dos resíduos (Real - Previsto)."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df_eval["residuo_lgbm"],
            name="Resíduo LightGBM (MWmed)",
            marker_color=COLOR_ERROR,
            opacity=0.72,
            nbinsx=40,
            hovertemplate=(
                "Faixa de Resíduo: %{x:,.0f} MWmed<br>Frequência: %{y} dias<extra></extra>"
            ),
        )
    )

    residuo_base = df_eval["carga_real_mwmed"] - df_eval["baseline_semanal_mwmed"]
    fig.add_trace(
        go.Histogram(
            x=residuo_base,
            name="Resíduo Baseline (MWmed)",
            marker_color=COLOR_BASELINE,
            opacity=0.45,
            nbinsx=40,
            hovertemplate=(
                "Faixa de Resíduo: %{x:,.0f} MWmed<br>Frequência: %{y} dias<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_width=1.4, line_dash="dash", line_color="#334155")

    fig.update_layout(
        template="plotly_white",
        title="Distribuição de Resíduos (Carga Real - Prevista)",
        barmode="overlay",
        height=380,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Resíduo (MWmed)", showgrid=False, tickformat=",.0f"),
        yaxis=dict(title="Contagem de Dias", showgrid=True, gridcolor=COLOR_GRID),
    )
    return fig


def build_folds_bar_figure(bt_folds: pd.DataFrame) -> go.Figure:
    """Constrói gráfico de barras comparando o MAE por janela temporal (fold)."""
    fig = go.Figure()

    fold_labels = [f"Fold {int(f)}" for f in bt_folds["fold"]]
    fig.add_trace(
        go.Bar(
            x=fold_labels,
            y=bt_folds["baseline_mae_mwmed"],
            name="Baseline Semanal",
            marker_color=COLOR_BASELINE,
            hovertemplate="%{x}<br>MAE Baseline: %{y:,.1f} MWmed<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=fold_labels,
            y=bt_folds["lgbm_mae_mwmed"],
            name="LightGBM D+7",
            marker_color=COLOR_FORECAST,
            hovertemplate="%{x}<br>MAE LightGBM: %{y:,.1f} MWmed<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="Desempenho Walk-Forward por Janela de 90 Dias (MAE em MWmed)",
        barmode="group",
        height=360,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Janela de Backtesting", showgrid=False),
        yaxis=dict(title="MAE (MWmed)", showgrid=True, gridcolor=COLOR_GRID, tickformat=",.0f"),
    )
    return fig


def build_climate_correlation_figure(df_filtered: pd.DataFrame) -> go.Figure:
    """Constrói gráfico de eixo duplo mostrando Carga Real (MWmed) e Temperatura Média SE (°C)."""
    clima_df = df_filtered.dropna(subset=["carga_real_mwmed", "temp_media_se"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if not clima_df.empty:
        fig.add_trace(
            go.Scatter(
                x=clima_df["data_referencia"],
                y=clima_df["carga_real_mwmed"],
                name="Carga Real (MWmed)",
                line=dict(color=COLOR_REAL, width=2.0),
                hovertemplate="Data: %{x|%d/%m/%Y}<br>Carga Real: %{y:,.1f} MWmed<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=clima_df["data_referencia"],
                y=clima_df["temp_media_se"],
                name="Temperatura Média Ponderada SE (°C)",
                line=dict(color=COLOR_TEMP, width=1.6, dash="dot"),
                hovertemplate=(
                    "Data: %{x|%d/%m/%Y}<br>Temperatura Média: %{y:.1f} °C<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        template="plotly_white",
        title="Dinâmica Carga Elétrica (MWmed) vs Temperatura Média SE (°C)",
        height=380,
        margin=dict(l=20, r=20, t=45, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title_text="Data de Referência", showgrid=False)
    fig.update_yaxes(
        title_text="Carga de Energia (MWmed)",
        secondary_y=False,
        showgrid=True,
        gridcolor=COLOR_GRID,
        tickformat=",.0f",
    )
    fig.update_yaxes(title_text="Temperatura Média (°C)", secondary_y=True, showgrid=False)
    return fig


def build_climate_scenarios_figure(clima_comp: pd.DataFrame) -> go.Figure:
    """Constrói gráfico comparativo dos 5 cenários experimentais com temperatura exógena."""
    colors = [
        COLOR_BASELINE,
        COLOR_REAL,
        "#17becf",
        COLOR_FORECAST,
        COLOR_TEMP,
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=clima_comp["mae_mwmed"],
                y=clima_comp["modelo"],
                orientation="h",
                marker_color=colors[: len(clima_comp)],
                text=[
                    f"{mae:,.1f} MWmed (MAPE {mape:.2f}%)"
                    for mae, mape in zip(
                        clima_comp["mae_mwmed"], clima_comp["mape_pct"], strict=True
                    )
                ],
                textposition="auto",
                hovertemplate="%{y}<br>MAE: %{x:,.1f} MWmed<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        title="Impacto da Temperatura Exógena (12 Janelas Walk-Forward)",
        height=380,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis=dict(title="MAE (MWmed)", showgrid=True, gridcolor=COLOR_GRID, tickformat=",.0f"),
        yaxis=dict(title="", autorange="reversed", showgrid=False),
    )
    return fig


def render_sidebar(
    df_timeline: pd.DataFrame,
    bt_folds: pd.DataFrame,
) -> Tuple[date, date, bool, bool, str, str]:
    """Renderiza os filtros interativos na barra lateral e retorna as seleções do usuário."""
    st.sidebar.header("Filtros de Análise")

    min_date = df_timeline["data_referencia"].min().date()
    max_date = df_timeline["data_referencia"].max().date()
    default_start = max(min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=90)).date())

    preset = st.sidebar.selectbox(
        "Recorte Temporal Rápido",
        options=[
            "Últimos 90 dias + Previsão D+7",
            "Últimos 30 dias + Previsão D+7",
            "Período de Backtesting (2024-2026)",
            "Série Completa (2018-2026)",
            "Personalizado",
        ],
        index=0,
    )

    if preset == "Últimos 30 dias + Previsão D+7":
        preset_start = max(min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=37)).date())
        preset_end = max_date
    elif preset == "Últimos 90 dias + Previsão D+7":
        preset_start = default_start
        preset_end = max_date
    elif preset == "Período de Backtesting (2024-2026)":
        preset_start = date(2024, 1, 1)
        preset_end = max_date
    elif preset == "Série Completa (2018-2026)":
        preset_start = min_date
        preset_end = max_date
    else:
        preset_start = default_start
        preset_end = max_date

    start_date = st.sidebar.date_input(
        "Data Inicial",
        value=preset_start,
        min_value=min_date,
        max_value=max_date,
    )
    end_date = st.sidebar.date_input(
        "Data Final",
        value=preset_end,
        min_value=min_date,
        max_value=max_date,
    )

    if start_date > end_date:
        st.sidebar.warning("A Data Inicial deve ser anterior ou igual à Data Final.")
        start_date, end_date = end_date, start_date

    st.sidebar.markdown("---")
    st.sidebar.subheader("Camadas do Gráfico")
    show_baseline = st.sidebar.checkbox("Exibir Baseline Semanal (t-7)", value=True)
    show_ci = st.sidebar.checkbox("Exibir Intervalo de Confiança (90%)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Segmentação ONS e Janelas")
    regime_filter = st.sidebar.selectbox(
        "Regime de Medição ONS",
        options=["Todos"] + list(REGIME_LABELS.keys()),
        format_func=lambda key: "Todos os Regimes" if key == "Todos" else REGIME_LABELS[key],
    )

    fold_options = ["Todos"] + [f"Fold {int(f)}" for f in bt_folds["fold"]]
    fold_filter = st.sidebar.selectbox("Janela de Backtesting (90 dias)", options=fold_options)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Fonte: Dados Abertos ONS (2018-2026) e Open-Meteo ERA5 | "
        "Armazenamento: DuckDB (Gold) | Rastreamento: MLflow"
    )

    return start_date, end_date, show_baseline, show_ci, regime_filter, fold_filter


def main() -> None:
    """Ponto de entrada principal da aplicação Streamlit."""
    st.set_page_config(
        page_title="Previsão de Carga de Energia - Subsistema SE (ONS)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_clean_style()

    df_timeline, df_forecast_7d, bt_folds, clima_comp, clima_folds, _ = load_dashboard_bundle()

    # Cabeçalho principal sem emojis ou símbolos decorativos
    st.title("Dashboard de Previsão de Demanda de Energia - Subsistema SE (ONS)")
    st.markdown(
        '<p class="dashboard-subtitle">'
        "Monitoramento operacional da carga diária de energia elétrica (MWmed) no Subsistema "
        "Sudeste/Centro-Oeste, comparando o modelo LightGBM (horizonte D+7 sem vazamento "
        "temporal) ao Baseline Ingênuo Semanal e analisando o ganho da temperatura exógena."
        "</p>",
        unsafe_allow_html=True,
    )

    start_date, end_date, show_baseline, show_ci, regime_filter, fold_filter = render_sidebar(
        df_timeline, bt_folds
    )

    # Aplicação dinâmica dos filtros selecionados
    mask = (df_timeline["data_referencia"].dt.date >= start_date) & (
        df_timeline["data_referencia"].dt.date <= end_date
    )
    df_filtered = df_timeline.loc[mask].copy()

    if regime_filter != "Todos":
        df_filtered = df_filtered[df_filtered["regime_medicao_ons"] == regime_filter]

    if fold_filter != "Todos":
        selected_fold = int(fold_filter.replace("Fold ", ""))
        df_filtered = df_filtered[df_filtered["fold"] == selected_fold]

    kpis = compute_kpis(df_filtered, df_forecast_7d)

    # Cartões de métricas principais no topo da página
    col1, col2, col3, col4 = st.columns(4)
    has_eval = int(kpis["n_eval"]) > 0

    with col1:
        if has_eval:
            diff_mape = float(kpis["lgbm_mape"]) - float(kpis["base_mape"])
            st.metric(
                label="MAPE (Erro Percentual Médio)",
                value=f"{_format_pt_number(float(kpis['lgbm_mape']), 2)}%",
                delta=f"{_format_pt_number(diff_mape, 2)} p.p. vs Baseline",
                delta_color="inverse",
            )
        else:
            st.metric(label="MAPE (Erro Percentual Médio)", value="N/D")

    with col2:
        if has_eval:
            diff_mae = float(kpis["lgbm_mae"]) - float(kpis["base_mae"])
            st.metric(
                label="MAE (Erro Absoluto Médio)",
                value=f"{_format_pt_number(float(kpis['lgbm_mae']), 1)} MWmed",
                delta=f"{_format_pt_number(diff_mae, 1)} MWmed vs Baseline",
                delta_color="inverse",
            )
        else:
            st.metric(label="MAE (Erro Absoluto Médio)", value="N/D")

    with col3:
        if has_eval:
            diff_rmse = float(kpis["lgbm_rmse"]) - float(kpis["base_rmse"])
            st.metric(
                label="RMSE (Raiz do Erro Quadrático)",
                value=f"{_format_pt_number(float(kpis['lgbm_rmse']), 1)} MWmed",
                delta=f"{_format_pt_number(diff_rmse, 1)} MWmed vs Baseline",
                delta_color="inverse",
            )
        else:
            st.metric(label="RMSE (Raiz do Erro Quadrático)", value="N/D")

    with col4:
        st.metric(
            label="Pico de Carga Prevista (D+1 a D+7)",
            value=f"{_format_pt_number(float(kpis['peak_next_7d_mwmed']), 1)} MWmed",
            delta=f"Data prevista: {kpis['peak_next_7d_date']}",
            delta_color="off",
        )

    if not has_eval:
        st.info(
            "O período ou segmento selecionado não possui dias de avaliação out-of-sample "
            "(o backtesting walk-forward inicia em 01/01/2024). As métricas de erro estão "
            "marcadas como N/D."
        )

    # Gráfico principal da série histórica e projeção futura
    st.subheader("Série Histórica Observada e Previsão de Carga (Horizonte D+7)")
    if df_filtered.empty:
        st.warning("Nenhum registro encontrado para a combinação de filtros selecionada.")
    else:
        main_fig = build_main_timeline_figure(df_filtered, show_baseline, show_ci)
        st.plotly_chart(main_fig, width="stretch")

    # Seção secundária em abas organizadas
    tab_erro, tab_clima, tab_dados = st.tabs(
        [
            "Análise de Erro e Resíduos",
            "Bônus: Impacto Climático (Temperatura SE)",
            "Projeção D+7 e Tabela de Janelas",
        ]
    )

    with tab_erro:
        eval_df = df_filtered.dropna(
            subset=["carga_real_mwmed", "carga_prevista_lgbm_mwmed"]
        ).copy()
        if eval_df.empty:
            st.info(
                "Selecione um intervalo a partir de 01/01/2024 para visualizar a evolução diária "
                "e a distribuição dos resíduos."
            )
        else:
            err_col1, err_col2 = st.columns(2)
            with err_col1:
                st.plotly_chart(build_error_timeseries_figure(eval_df), width="stretch")
            with err_col2:
                st.plotly_chart(build_residual_histogram_figure(eval_df), width="stretch")

        st.plotly_chart(build_folds_bar_figure(bt_folds), width="stretch")

    with tab_clima:
        st.markdown(
            "O experimento bônus avalia a inclusão da temperatura média diária ponderada das "
            "quatro principais capitais do subsistema SE/CO (SP 50%, RJ 25%, BH 15%, DF 10%) "
            "proveniente da reanálise ERA5 (Open-Meteo). Com erro realista de previsão "
            "meteorológica D+7 (desvio padrão de 1,5 °C), o ganho de MAE sobre o baseline "
            "sobe de 11,38% para 24,61% e o modelo passa a vencer todas as 12 janelas."
        )
        clima_col1, clima_col2 = st.columns(2)
        with clima_col1:
            st.plotly_chart(build_climate_correlation_figure(df_filtered), width="stretch")
        with clima_col2:
            st.plotly_chart(build_climate_scenarios_figure(clima_comp), width="stretch")

        st.subheader("Comparativo Consolidado dos Cenários Climáticos")
        clima_display = clima_comp.rename(
            columns={
                "modelo": "Modelo / Cenário",
                "n_features": "Atributos",
                "mae_mwmed": "MAE (MWmed)",
                "mape_pct": "MAPE (%)",
                "rmse_mwmed": "RMSE (MWmed)",
                "ganho_mae_vs_base_pct": "Ganho MAE vs Baseline (%)",
                "ganho_mape_vs_base_pct": "Ganho MAPE vs Baseline (%)",
                "folds_vencidos": "Janelas Vencidas",
            }
        )
        st.dataframe(clima_display, width="stretch", hide_index=True)

    with tab_dados:
        st.subheader("Previsão Operacional para os Próximos 7 Dias (D+1 a D+7)")
        df_7d_display = df_forecast_7d.copy()
        df_7d_display["data_referencia"] = pd.to_datetime(
            df_7d_display["data_referencia"]
        ).dt.strftime("%Y-%m-%d")
        df_7d_display = df_7d_display.rename(
            columns={
                "data_referencia": "Data Prevista",
                "y_pred_lgbm": "Previsão LightGBM (MWmed)",
                "y_pred_baseline": "Baseline Semanal (MWmed)",
                "ic_inferior_mwmed": "IC 90% Inferior (MWmed)",
                "ic_superior_mwmed": "IC 90% Superior (MWmed)",
            }
        )
        st.dataframe(
            df_7d_display.style.format(
                {
                    "Previsão LightGBM (MWmed)": "{:,.1f}",
                    "Baseline Semanal (MWmed)": "{:,.1f}",
                    "IC 90% Inferior (MWmed)": "{:,.1f}",
                    "IC 90% Superior (MWmed)": "{:,.1f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            label="Exportar Previsão D+7 (CSV)",
            data=df_forecast_7d.to_csv(index=False).encode("utf-8"),
            file_name="previsao_carga_se_d7.csv",
            mime="text/csv",
        )

        st.subheader("Resumo por Janela do Backtesting Walk-Forward (Etapa 06)")
        folds_display = bt_folds.rename(
            columns={
                "fold": "Janela",
                "test_start": "Início Teste",
                "test_end": "Fim Teste",
                "n_dias": "Dias",
                "train_rows": "Linhas Treino",
                "baseline_mae_mwmed": "MAE Baseline (MWmed)",
                "lgbm_mae_mwmed": "MAE LightGBM (MWmed)",
                "mae_gain_pct": "Ganho MAE (%)",
                "baseline_mape_pct": "MAPE Baseline (%)",
                "lgbm_mape_pct": "MAPE LightGBM (%)",
                "mape_gain_pct": "Ganho MAPE (%)",
            }
        )
        st.dataframe(folds_display, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
