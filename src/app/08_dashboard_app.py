#!/usr/bin/env python3
"""08_dashboard_app.py

Interface interativa em Streamlit e Plotly (tema escuro corporativo) para apresentação
dos resultados do modelo de Previsão de Carga Diária de Energia (horizonte D+7) do
Subsistema Sudeste/Centro-Oeste (ONS), com filtros no topo acima dos cards de KPI e
tooltips explicativos ao passar o mouse sobre siglas e termos técnicos.
"""

from __future__ import annotations

import html
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

# Paleta escura corporativa alinhada ao layout visual da aplicação
COLOR_BG = "#0b0e14"
COLOR_CARD = "#141820"
COLOR_BORDER = "#262b36"
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"
COLOR_GRID = "#222834"

COLOR_REAL = "#1f77b4"
COLOR_FORECAST = "#ff7f0e"
COLOR_CI_FILL = "rgba(255, 127, 14, 0.18)"
COLOR_BASELINE = "#8b949e"
COLOR_ERROR = "#ef4444"
COLOR_TEMP = "#2ca02c"

# Glossário de siglas e termos técnicos exibidos ao passar o mouse
GLOSSARY: Dict[str, str] = {
    "SE": (
        "Subsistema Sudeste/Centro-Oeste do Sistema Interligado Nacional (SIN), "
        "responsável pela maior parcela da carga elétrica brasileira."
    ),
    "ONS": (
        "Operador Nacional do Sistema Elétrico: órgão responsável por coordenar e controlar "
        "a operação da geração e transmissão de energia no Brasil."
    ),
    "MWmed": (
        "Megawatt médio: potência média verificada no dia (energia diária em MWh "
        "dividida pelas 24 horas), unidade oficial de carga do ONS."
    ),
    "LightGBM": (
        "Light Gradient Boosting Machine: algoritmo de aprendizado de máquina baseado em "
        "árvores de decisão de alta eficiência para dados tabulares."
    ),
    "D+7": (
        "Horizonte de previsão de até 7 dias à frente a partir da última data "
        "histórica consolidada."
    ),
    "D+1 a D+7": (
        "Janela operacional dos próximos 7 dias consecutivos após o último dado real observado."
    ),
    "Baseline Ingênuo Semanal": (
        "Modelo de referência que repete o valor de carga observado exatamente 7 dias antes "
        "(mesmo dia da semana anterior)."
    ),
    "t-7": ("Defasagem temporal de 7 dias atrás: repete a carga do mesmo dia da semana anterior."),
    "Open-Meteo ERA5": (
        "Base climática global de reanálise atmosférica (ECMWF) utilizada para obter "
        "séries diárias contínuas de temperatura sem falhas."
    ),
    "DuckDB Gold": (
        "Camada analítica final armazenada em DuckDB com dados tratados, contínuos e "
        "validados pelo dbt para modelagem e consumo."
    ),
    "MAPE": (
        "Mean Absolute Percentage Error (Erro Percentual Absoluto Médio): média das "
        "diferenças percentuais absolutas entre carga real e prevista."
    ),
    "MAE": (
        "Mean Absolute Error (Erro Absoluto Médio): média dos erros absolutos em MWmed, "
        "indicando o desvio médio típico sem anulação de sinais."
    ),
    "RMSE": (
        "Root Mean Squared Error (Raiz do Erro Quadrático Médio): métrica em MWmed que "
        "penaliza com maior peso grandes desvios pontuais."
    ),
    "IC 90%": (
        "Intervalo de Confiança empírico de 90%: faixa estimada pelos quantis P5 e P95 "
        "dos erros históricos fora da amostra."
    ),
    "MMGD": (
        "Micro e Minigeração Distribuída (ex.: geração solar em telhados), incorporada "
        "pelo ONS à carga total a partir de 29/04/2023."
    ),
    "Backtesting": (
        "Validação histórica em janelas sequenciais (walk-forward) simulando previsões em "
        "produção sem vazamento de dados futuros."
    ),
    "Fold": (
        "Cada janela temporal independente de 90 dias utilizada na avaliação "
        "walk-forward do modelo."
    ),
    "Resíduos": (
        "Diferença entre a carga real observada e a carga prevista pelo modelo "
        "(Real - Previsto), em MWmed."
    ),
}

REGIME_LABELS: Dict[str, str] = {
    "1_despachada": "Regime 1: Despachadas (até fev/2021)",
    "2_nao_despachada": "Regime 2: Não Despachadas (2021-2023)",
    "3_com_mmgd": "Regime 3: Com MMGD (a partir abr/2023)",
}

REGIME_TRANSITIONS = [
    ("2021-03-01", "ONS Não Despachadas"),
    ("2023-04-29", "ONS Inclusão MMGD"),
]


def _term(label: str, key: str | None = None) -> str:
    """Envolve uma sigla ou termo técnico com tooltip CSS exibido ao passar o mouse."""
    tip_text = GLOSSARY.get(key or label, "")
    if not tip_text:
        return html.escape(label)
    safe_tip = html.escape(tip_text, quote=True)
    safe_label = html.escape(label)
    return f'<span class="term-tip" data-tip="{safe_tip}" tabindex="0">{safe_label}</span>'


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


def _apply_dark_style() -> None:
    """Aplica tema escuro corporativo fiel ao layout visual e estiliza tooltips de termos."""
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {COLOR_BG};
                color: {COLOR_TEXT};
            }}
            .block-container {{
                padding-top: 1.75rem;
                padding-bottom: 3rem;
                max-width: 1400px;
            }}
            header[data-testid="stHeader"] {{
                background-color: transparent;
            }}
            /* Painel de filtros e cards no estilo escuro */
            div[data-testid="stVerticalBlockBorderWrapper"] {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                overflow: visible;
            }}
            div[data-testid="stMetric"] {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                padding: 1.05rem 1.15rem;
                min-height: 132px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }}
            div[data-testid="stMetricLabel"] {{
                font-weight: 500;
                color: {COLOR_MUTED};
                font-size: 0.85rem;
                line-height: 1.38;
                margin-bottom: 0.35rem;
            }}
            div[data-testid="stMetricValue"],
            div[data-testid="stMetricValue"] > div {{
                font-weight: 700;
                color: {COLOR_TEXT};
                font-size: clamp(1.28rem, 1.75vw, 1.72rem);
                line-height: 1.25;
                white-space: nowrap;
            }}
            div[data-testid="stMetricDelta"] {{
                margin-top: 0.35rem;
                font-size: 0.82rem;
                line-height: 1.35;
            }}
            /* Abas e containers secundários com espaçamento confortável */
            div[data-testid="stTabs"] {{
                margin-top: 0.65rem;
            }}
            div[data-testid="stTabs"] button[role="tab"] {{
                padding-top: 0.65rem;
                padding-bottom: 0.65rem;
                font-size: 0.92rem;
            }}
            div[data-testid="stTabs"] div[data-baseweb="tab-panel"] {{
                padding-top: 1.1rem;
            }}
            .dashboard-header {{
                border-bottom: 1px solid {COLOR_BORDER};
                padding-bottom: 1.1rem;
                margin-bottom: 1.25rem;
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                align-items: flex-end;
                gap: 0.9rem 1.5rem;
            }}
            .dashboard-title {{
                font-size: clamp(1.42rem, 2.05vw, 1.75rem);
                font-weight: 700;
                color: {COLOR_TEXT};
                margin: 0;
                line-height: 1.32;
            }}
            .dashboard-subtitle {{
                color: {COLOR_MUTED};
                font-size: 0.93rem;
                margin: 0.5rem 0 0 0;
                max-width: 840px;
                line-height: 1.58;
            }}
            .dashboard-source {{
                color: {COLOR_MUTED};
                font-size: 0.78rem;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                line-height: 1.45;
            }}
            /* Tooltips interativos para siglas e termos técnicos */
            .term-tip {{
                position: relative;
                display: inline-block;
                border-bottom: 1px dotted #64748b;
                cursor: help;
                color: inherit;
            }}
            .term-tip:hover,
            .term-tip:focus {{
                border-bottom-color: {COLOR_FORECAST};
                outline: none;
            }}
            .term-tip::after {{
                content: attr(data-tip);
                position: absolute;
                left: 50%;
                bottom: calc(100% + 8px);
                transform: translateX(-50%);
                width: max-content;
                max-width: 290px;
                white-space: normal;
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 0.55rem 0.75rem;
                font-size: 0.76rem;
                font-weight: 400;
                line-height: 1.45;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.45);
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.12s ease;
                z-index: 9999;
                text-align: left;
            }}
            .term-tip:hover::after,
            .term-tip:focus::after {{
                opacity: 1;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_pt_number(value: float, decimals: int = 1) -> str:
    """Formata números com separador de milhar em ponto e decimal em vírgula."""
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def _base_dark_layout(
    fig: go.Figure,
    title: str = "",
    height: int = 400,
    top_margin: int | None = None,
) -> None:
    """Aplica padronização visual escura limpa e sem colisão entre título e legenda."""
    resolved_top = top_margin if top_margin is not None else (88 if title else 52)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor=COLOR_CARD,
        font=dict(color=COLOR_TEXT, family="sans-serif", size=12),
        title=(
            dict(
                text=title,
                x=0.01,
                xanchor="left",
                y=0.96,
                yanchor="top",
                font=dict(size=14, color=COLOR_TEXT),
            )
            if title
            else None
        ),
        height=height,
        margin=dict(l=24, r=24, t=resolved_top, b=36),
    )


def build_main_timeline_figure(
    df_filtered: pd.DataFrame,
    show_baseline: bool,
    show_ci: bool,
) -> go.Figure:
    """Constrói o gráfico principal de série histórica, backtesting out-of-sample e previsão D+7."""
    fig = go.Figure()

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
                name="IC 90%",
                hovertemplate=(
                    "Data: %{x|%d/%m/%Y}<br>Limite Inferior (90%): %{y:,.1f} MWmed<extra></extra>"
                ),
            )
        )

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

    pred_df = df_filtered.dropna(subset=["carga_prevista_lgbm_mwmed"])
    if not pred_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pred_df["data_referencia"],
                y=pred_df["carga_prevista_lgbm_mwmed"],
                mode="lines",
                name="Previsão LightGBM (D+7)",
                line=dict(color=COLOR_FORECAST, width=2.2, dash="dash"),
                hovertemplate=(
                    "Data: %{x|%d/%m/%Y}<br>Previsão LightGBM: %{y:,.1f} MWmed<extra></extra>"
                ),
            )
        )

    base_df = df_filtered.dropna(subset=["baseline_semanal_mwmed"])
    if show_baseline and not base_df.empty:
        fig.add_trace(
            go.Scatter(
                x=base_df["data_referencia"],
                y=base_df["baseline_semanal_mwmed"],
                mode="lines",
                name="Baseline Semanal",
                line=dict(color=COLOR_BASELINE, width=1.5, dash="dot"),
                opacity=0.85,
                hovertemplate=(
                    "Data: %{x|%d/%m/%Y}<br>Baseline Semanal: %{y:,.1f} MWmed<extra></extra>"
                ),
            )
        )

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
                    line_color="#64748b",
                    annotation_text=label,
                    annotation_position="top left",
                    annotation_font_size=11,
                    annotation_font_color=COLOR_MUTED,
                    annotation_yshift=-4,
                )

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
                annotation_yshift=-4,
            )

    y_cols = ["carga_real_mwmed", "carga_prevista_lgbm_mwmed"]
    if show_baseline:
        y_cols.append("baseline_semanal_mwmed")
    if show_ci:
        y_cols.extend(["ic_inferior_mwmed", "ic_superior_mwmed"])
    y_vals = pd.concat([df_filtered[c].dropna() for c in y_cols if c in df_filtered.columns])
    yaxis_range = None
    if not y_vals.empty:
        y_min = float(y_vals.min())
        y_max = float(y_vals.max())
        span = max(y_max - y_min, 1000.0)
        yaxis_range = [y_min - span * 0.05, y_max + span * 0.14]

    _base_dark_layout(fig, height=465, top_margin=58)
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_MUTED, size=12),
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            showline=True,
            linecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_MUTED),
            automargin=True,
        ),
        yaxis=dict(
            title="Carga de Energia (MWmed)",
            range=yaxis_range,
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            tickformat=",.0f",
            tickfont=dict(color=COLOR_MUTED),
            title_font=dict(color=COLOR_MUTED),
            title_standoff=10,
            automargin=True,
        ),
    )
    return fig


def build_error_timeseries_figure(df_eval: pd.DataFrame) -> go.Figure:
    """Constrói gráfico de área do erro absoluto diário destacando picos de erro."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_eval["data_referencia"],
            y=df_eval["erro_absoluto_lgbm"],
            mode="lines",
            name="Erro Absoluto LightGBM",
            line=dict(color=COLOR_ERROR, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.16)",
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
            opacity=0.85,
            hovertemplate=("Data: %{x|%d/%m/%Y}<br>Erro Baseline: %{y:,.1f} MWmed<extra></extra>"),
        )
    )

    _base_dark_layout(
        fig,
        title="Evolução Diária do Erro Absoluto (MWmed)",
        height=395,
        top_margin=92,
    )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            x=0,
            font=dict(color=COLOR_MUTED, size=11.5),
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            linecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_MUTED),
            automargin=True,
        ),
        yaxis=dict(
            title="Erro Absoluto (MWmed)",
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat=",.0f",
            tickfont=dict(color=COLOR_MUTED),
            title_standoff=10,
            automargin=True,
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

    fig.add_vline(x=0, line_width=1.4, line_dash="dash", line_color=COLOR_MUTED)

    _base_dark_layout(
        fig,
        title="Distribuição de Resíduos (Carga Real - Prevista)",
        height=395,
        top_margin=92,
    )
    fig.update_layout(
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            x=0,
            font=dict(color=COLOR_MUTED, size=11.5),
        ),
        xaxis=dict(
            title="Resíduo (MWmed)",
            showgrid=False,
            tickformat=",.0f",
            tickfont=dict(color=COLOR_MUTED),
            title_standoff=10,
            automargin=True,
        ),
        yaxis=dict(
            title="Contagem de Dias",
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickfont=dict(color=COLOR_MUTED),
            title_standoff=10,
            automargin=True,
        ),
    )
    return fig


def build_folds_bar_figure(bt_folds: pd.DataFrame) -> go.Figure:
    """Constrói gráfico de barras comparando o MAE por janela temporal (fold)."""
    fig = go.Figure()

    fold_labels = [f"F{int(f)}" for f in bt_folds["fold"]]
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

    _base_dark_layout(
        fig,
        title="Desempenho Walk-Forward por Janela de 90 Dias (MAE em MWmed)",
        height=385,
        top_margin=92,
    )
    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.10,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            x=0,
            font=dict(color=COLOR_MUTED, size=11.5),
        ),
        xaxis=dict(
            title="Janela de Backtesting",
            showgrid=False,
            tickfont=dict(color=COLOR_MUTED),
            title_standoff=10,
            automargin=True,
        ),
        yaxis=dict(
            title="MAE (MWmed)",
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat=",.0f",
            tickfont=dict(color=COLOR_MUTED),
            title_standoff=10,
            automargin=True,
        ),
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

    _base_dark_layout(
        fig,
        title="Dinâmica Carga Elétrica (MWmed) vs Temperatura Média SE (°C)",
        height=395,
        top_margin=92,
    )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            x=0,
            font=dict(color=COLOR_MUTED, size=11.5),
        ),
    )
    fig.update_xaxes(
        title_text="", showgrid=False, tickfont=dict(color=COLOR_MUTED), automargin=True
    )
    fig.update_yaxes(
        title_text="Carga de Energia (MWmed)",
        secondary_y=False,
        showgrid=True,
        gridcolor=COLOR_GRID,
        tickformat=",.0f",
        tickfont=dict(color=COLOR_MUTED),
        title_standoff=10,
        automargin=True,
    )
    fig.update_yaxes(
        title_text="Temperatura Média (°C)",
        secondary_y=True,
        showgrid=False,
        tickfont=dict(color=COLOR_MUTED),
        title_standoff=10,
        automargin=True,
    )
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
    max_mae = float(clima_comp["mae_mwmed"].max()) if not clima_comp.empty else 2500.0
    fig = go.Figure(
        data=[
            go.Bar(
                x=clima_comp["mae_mwmed"],
                y=clima_comp["modelo"],
                orientation="h",
                marker_color=colors[: len(clima_comp)],
                text=[
                    f"{_format_pt_number(mae, 1)} MWmed (MAPE {_format_pt_number(mape, 2)}%)"
                    for mae, mape in zip(
                        clima_comp["mae_mwmed"], clima_comp["mape_pct"], strict=True
                    )
                ],
                textposition="inside",
                insidetextanchor="end",
                textfont=dict(color=COLOR_TEXT, size=11.5),
                hovertemplate="%{y}<br>MAE: %{x:,.1f} MWmed<extra></extra>",
            )
        ]
    )
    _base_dark_layout(
        fig,
        title="Impacto da Temperatura Exógena (12 Janelas Walk-Forward)",
        height=395,
        top_margin=64,
    )
    fig.update_layout(
        bargap=0.26,
        xaxis=dict(
            title="MAE (MWmed)",
            range=[0, max_mae * 1.12],
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat=",.0f",
            tickfont=dict(color=COLOR_MUTED),
            title_standoff=10,
            automargin=True,
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            showgrid=False,
            tickfont=dict(color=COLOR_MUTED, size=11),
            automargin=True,
        ),
    )
    return fig


def render_top_filters(
    df_timeline: pd.DataFrame,
    bt_folds: pd.DataFrame,
) -> Tuple[date, date, bool, bool, str, str]:
    """Renderiza a barra horizontal de filtros em card acima dos KPIs e retorna as seleções."""
    min_date = df_timeline["data_referencia"].min().date()
    max_date = df_timeline["data_referencia"].max().date()
    default_start = max(min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=90)).date())

    preset_options = [
        "Últimos 90 dias + Previsão D+7",
        "Últimos 30 dias + Previsão D+7",
        "Período de Backtesting (2024-2026)",
        "Série Completa (2018-2026)",
        "Personalizado",
    ]

    if "filter_start_date" not in st.session_state:
        st.session_state["filter_start_date"] = default_start
    if "filter_end_date" not in st.session_state:
        st.session_state["filter_end_date"] = max_date
    if "filter_preset" not in st.session_state:
        st.session_state["filter_preset"] = preset_options[0]

    def _on_preset_change() -> None:
        selected = st.session_state["filter_preset"]
        if selected == "Últimos 30 dias + Previsão D+7":
            st.session_state["filter_start_date"] = max(
                min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=37)).date()
            )
            st.session_state["filter_end_date"] = max_date
        elif selected == "Últimos 90 dias + Previsão D+7":
            st.session_state["filter_start_date"] = default_start
            st.session_state["filter_end_date"] = max_date
        elif selected == "Período de Backtesting (2024-2026)":
            st.session_state["filter_start_date"] = date(2024, 1, 1)
            st.session_state["filter_end_date"] = max_date
        elif selected == "Série Completa (2018-2026)":
            st.session_state["filter_start_date"] = min_date
            st.session_state["filter_end_date"] = max_date

    def _on_manual_date_change() -> None:
        st.session_state["filter_preset"] = "Personalizado"

    with st.container(border=True):
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns([1.65, 1.0, 1.0, 1.65], gap="medium")

        with r1_c1:
            st.selectbox(
                "Recorte Temporal",
                options=preset_options,
                key="filter_preset",
                on_change=_on_preset_change,
                help=(
                    "Atalhos rápidos de período. A projeção D+7 cobre os 7 dias "
                    "seguintes ao último registro histórico observado."
                ),
            )
        with r1_c2:
            start_date = st.date_input(
                "Data Inicial",
                min_value=min_date,
                max_value=max_date,
                key="filter_start_date",
                on_change=_on_manual_date_change,
                help="Data de início exibida na série temporal e usada nos KPIs.",
            )
        with r1_c3:
            end_date = st.date_input(
                "Data Final",
                min_value=min_date,
                max_value=max_date,
                key="filter_end_date",
                on_change=_on_manual_date_change,
                help="Data final exibida (inclui o horizonte projetado D+1 a D+7).",
            )
        with r1_c4:
            regime_filter = st.selectbox(
                "Regime de Medição ONS",
                options=["Todos"] + list(REGIME_LABELS.keys()),
                format_func=lambda key: (
                    "Todos os Regimes" if key == "Todos" else REGIME_LABELS[key]
                ),
                help=(
                    "Marcos metodológicos do ONS: inclusão de geração não despachada "
                    "(mar/2021) e estimativa de MMGD - Micro e Minigeração Distribuída (abr/2023)."
                ),
            )

        r2_c1, r2_c2, r2_c3 = st.columns([1.65, 1.85, 1.80], gap="medium")
        with r2_c1:
            fold_options = ["Todos"] + [f"Fold {int(f)}" for f in bt_folds["fold"]]
            fold_filter = st.selectbox(
                "Janela Backtesting (90d)",
                options=fold_options,
                format_func=lambda v: "Todas as Janelas" if v == "Todos" else v,
                help=(
                    "Cada Fold representa uma janela independente de 90 dias avaliada fora "
                    "da amostra no backtesting walk-forward (2024 a 2026)."
                ),
            )
        with r2_c2:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            show_baseline = st.checkbox(
                "Exibir Baseline Semanal (t-7)",
                value=True,
                help=GLOSSARY["Baseline Ingênuo Semanal"],
            )
        with r2_c3:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            show_ci = st.checkbox(
                "Exibir Intervalo de Confiança (90%)",
                value=True,
                help=GLOSSARY["IC 90%"],
            )

    if start_date > end_date:
        st.warning("A Data Inicial deve ser anterior ou igual à Data Final.")
        start_date, end_date = end_date, start_date

    return start_date, end_date, show_baseline, show_ci, regime_filter, fold_filter


def main() -> None:
    """Ponto de entrada principal da aplicação Streamlit."""
    st.set_page_config(
        page_title="Dashboard de Previsão de Demanda de Energia - Subsistema SE (ONS)",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_dark_style()

    df_timeline, df_forecast_7d, bt_folds, clima_comp, _, _ = load_dashboard_bundle()

    # Cabeçalho superior com tooltips explicativos nas siglas
    header_html = (
        '<div class="dashboard-header">'
        "<div>"
        '<h1 class="dashboard-title">'
        "Dashboard de Previsão de Demanda de Energia - "
        f'<span style="white-space:nowrap;">Subsistema {_term("SE")} ({_term("ONS")})</span>'
        "</h1>"
        '<p class="dashboard-subtitle">'
        f"Monitoramento operacional da carga diária de energia elétrica ({_term('MWmed')}) "
        f"no Subsistema Sudeste/Centro-Oeste, comparando o modelo {_term('LightGBM')} "
        f"(horizonte {_term('D+7')} sem vazamento temporal) ao "
        f"{_term('Baseline Ingênuo Semanal')} e ao experimento climático."
        "</p>"
        "</div>"
        '<div class="dashboard-source">'
        f"Fonte: {_term('ONS')} (2018-2026) | {_term('Open-Meteo ERA5')} | {_term('DuckDB Gold')}"
        "</div>"
        "</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # Filtros posicionados no topo acima dos cartões de KPI
    start_date, end_date, show_baseline, show_ci, regime_filter, fold_filter = render_top_filters(
        df_timeline, bt_folds
    )

    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

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

    # Cartões de métricas principais logo abaixo dos filtros
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    has_eval = int(kpis["n_eval"]) > 0

    with col1:
        if has_eval:
            diff_mape = float(kpis["lgbm_mape"]) - float(kpis["base_mape"])
            st.metric(
                label="MAPE (Erro Percentual Médio)",
                value=f"{_format_pt_number(float(kpis['lgbm_mape']), 2)}%",
                delta=f"{_format_pt_number(diff_mape, 2)} p.p. vs Baseline",
                delta_color="inverse",
                help=GLOSSARY["MAPE"],
            )
        else:
            st.metric(
                label="MAPE (Erro Percentual Médio)",
                value="N/D",
                help=GLOSSARY["MAPE"],
            )

    with col2:
        if has_eval:
            diff_mae = float(kpis["lgbm_mae"]) - float(kpis["base_mae"])
            st.metric(
                label="MAE (Erro Absoluto Médio)",
                value=f"{_format_pt_number(float(kpis['lgbm_mae']), 1)} MWmed",
                delta=f"{_format_pt_number(diff_mae, 1)} MWmed vs Baseline",
                delta_color="inverse",
                help=f"{GLOSSARY['MAE']} | {GLOSSARY['MWmed']}",
            )
        else:
            st.metric(
                label="MAE (Erro Absoluto Médio)",
                value="N/D",
                help=f"{GLOSSARY['MAE']} | {GLOSSARY['MWmed']}",
            )

    with col3:
        if has_eval:
            diff_rmse = float(kpis["lgbm_rmse"]) - float(kpis["base_rmse"])
            st.metric(
                label="RMSE (Raiz do Erro Quadrático)",
                value=f"{_format_pt_number(float(kpis['lgbm_rmse']), 1)} MWmed",
                delta=f"{_format_pt_number(diff_rmse, 1)} MWmed vs Baseline",
                delta_color="inverse",
                help=f"{GLOSSARY['RMSE']} | {GLOSSARY['MWmed']}",
            )
        else:
            st.metric(
                label="RMSE (Raiz do Erro Quadrático)",
                value="N/D",
                help=f"{GLOSSARY['RMSE']} | {GLOSSARY['MWmed']}",
            )

    with col4:
        peak_date_br = pd.Timestamp(str(kpis["peak_next_7d_date"])).strftime("%d/%m/%Y")
        st.metric(
            label="Pico Previsto (D+1 a D+7)",
            value=f"{_format_pt_number(float(kpis['peak_next_7d_mwmed']), 1)} MWmed",
            delta=f"Data prevista: {peak_date_br}",
            delta_color="off",
            help=f"{GLOSSARY['D+1 a D+7']} | {GLOSSARY['MWmed']}",
        )

    if not has_eval:
        st.info(
            "O período ou segmento selecionado não possui dias de avaliação out-of-sample "
            "(o backtesting walk-forward inicia em 01/01/2024). As métricas de erro estão "
            "marcadas como N/D."
        )

    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

    # Gráfico principal dentro de card escuro
    with st.container(border=True):
        st.markdown(
            f"<div style='font-size: 1.05rem; font-weight: 600; "
            f"line-height: 1.45; margin-bottom: 0.35rem;'>"
            f"Série Histórica Observada e Previsão de Carga (Horizonte {_term('D+7')}) "
            f"&nbsp;|&nbsp; <span style='font-size:0.82rem;font-weight:400;color:{COLOR_MUTED};'>"
            f"{_term('IC 90%')} &middot; {_term('t-7')} &middot; {_term('MMGD')}"
            f"</span></div>",
            unsafe_allow_html=True,
        )
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
        st.markdown(
            f"<p style='font-size:0.87rem;color:{COLOR_MUTED};"
            f"line-height:1.55;margin-bottom:0.95rem;'>"
            f"Avaliação de {_term('Resíduos')} e erros diários ({_term('MAE')} em "
            f"{_term('MWmed')}) ao longo das janelas de {_term('Backtesting')} "
            f"({_term('Fold')} de 90 dias).</p>",
            unsafe_allow_html=True,
        )
        eval_df = df_filtered.dropna(
            subset=["carga_real_mwmed", "carga_prevista_lgbm_mwmed"]
        ).copy()
        if eval_df.empty:
            st.info(
                "Selecione um intervalo a partir de 01/01/2024 para visualizar a evolução diária "
                "e a distribuição dos resíduos."
            )
        else:
            err_col1, err_col2 = st.columns(2, gap="medium")
            with err_col1:
                st.plotly_chart(build_error_timeseries_figure(eval_df), width="stretch")
            with err_col2:
                st.plotly_chart(build_residual_histogram_figure(eval_df), width="stretch")

        st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)
        st.plotly_chart(build_folds_bar_figure(bt_folds), width="stretch")

    with tab_clima:
        st.markdown(
            f"<p style='font-size:0.87rem;color:{COLOR_MUTED};"
            f"line-height:1.6;margin-bottom:0.95rem;'>"
            f"O experimento bônus avalia a inclusão da temperatura média diária ponderada "
            f"das quatro principais capitais do subsistema {_term('SE')} (SP 50%, RJ 25%, "
            f"BH 15%, DF 10%) via {_term('Open-Meteo ERA5')}. Com erro realista de previsão "
            f"meteorológica {_term('D+7')} (desvio padrão de 1,5 °C), o ganho de "
            f"{_term('MAE')} sobre o {_term('Baseline Ingênuo Semanal')} sobe de 11,38% "
            f"para 24,61% e o {_term('LightGBM')} passa a vencer todas as 12 janelas.</p>",
            unsafe_allow_html=True,
        )
        clima_col1, clima_col2 = st.columns(2, gap="medium")
        with clima_col1:
            st.plotly_chart(build_climate_correlation_figure(df_filtered), width="stretch")
        with clima_col2:
            st.plotly_chart(build_climate_scenarios_figure(clima_comp), width="stretch")

        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
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
        st.dataframe(
            clima_display.style.format(
                {
                    "MAE (MWmed)": "{:,.1f}",
                    "MAPE (%)": "{:.2f}",
                    "RMSE (MWmed)": "{:,.1f}",
                    "Ganho MAE vs Baseline (%)": "{:+.2f}",
                    "Ganho MAPE vs Baseline (%)": "{:+.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with tab_dados:
        st.subheader("Previsão Operacional para os Próximos 7 Dias (D+1 a D+7)")
        df_7d_display = df_forecast_7d.copy()
        df_7d_display["data_referencia"] = pd.to_datetime(
            df_7d_display["data_referencia"]
        ).dt.strftime("%d/%m/%Y")
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

        st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="Exportar Previsão D+7 (CSV)",
            data=df_forecast_7d.to_csv(index=False).encode("utf-8"),
            file_name="previsao_carga_se_d7.csv",
            mime="text/csv",
        )

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
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
        st.dataframe(
            folds_display.style.format(
                {
                    "MAE Baseline (MWmed)": "{:,.1f}",
                    "MAE LightGBM (MWmed)": "{:,.1f}",
                    "Ganho MAE (%)": "{:+.2f}",
                    "MAPE Baseline (%)": "{:.2f}",
                    "MAPE LightGBM (%)": "{:.2f}",
                    "Ganho MAPE (%)": "{:+.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
