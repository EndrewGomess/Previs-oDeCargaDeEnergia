#!/usr/bin/env python3
"""08_dashboard_slides.py

Aplicação Streamlit dedicada para apresentação executiva em 7 slides explicando por que o
modelo LightGBM supera o Baseline Ingênuo Semanal e exatamente onde cada bloco de
engenharia de atributos (lags multi-horizonte, estatísticas móveis, feriados, regime
ONS e temperatura exógena) gera ganho mensurável fora da amostra.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

COLOR_BG = "#0b0e14"
COLOR_CARD = "#141820"
COLOR_BORDER = "#262b36"
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"
COLOR_GRID = "#222834"

COLOR_REAL = "#1f77b4"
COLOR_FORECAST = "#ff7f0e"
COLOR_BASELINE = "#8b949e"
COLOR_ERROR = "#ef4444"
COLOR_TEMP = "#2ca02c"
COLOR_CYAN = "#17becf"

SLIDE_DEFINITIONS: List[Dict[str, str]] = [
    {
        "id": "tese",
        "nav_title": "Tese Central e Evolução de Desempenho",
        "tag": "SLIDE 01 / 07  |  TESE CENTRAL",
    },
    {
        "id": "anatomia_dias",
        "nav_title": "Onde o Modelo Vence: Anatomia por Tipo de Dia",
        "tag": "SLIDE 02 / 07  |  ONDE O GANHO ACONTECE",
    },
    {
        "id": "eco_feriados",
        "nav_title": "Caso Real 1: Feriados e o Efeito Eco D+7",
        "tag": "SLIDE 03 / 07  |  EVIDÊNCIA EM FERIADOS",
    },
    {
        "id": "choques_transitorios",
        "nav_title": "Caso Real 2: Amortecimento de Choques Transitórios",
        "tag": "SLIDE 04 / 07  |  ESTABILIDADE TEMPORAL",
    },
    {
        "id": "importancia_features",
        "nav_title": "O Que Mudou por Dentro: Grupos de Atributos",
        "tag": "SLIDE 05 / 07  |  ENGENHARIA DE ATRIBUTOS",
    },
    {
        "id": "fold5_clima",
        "nav_title": "Ponto Cego Sem Clima (Fold 5) e Virada com Temperatura",
        "tag": "SLIDE 06 / 07  |  IMPACTO CLIMÁTICO D+7",
    },
    {
        "id": "sintese_cauda",
        "nav_title": "Redução de Erros Extremos e Síntese Executiva",
        "tag": "SLIDE 07 / 07  |  SÍNTESE E PRODUÇÃO",
    },
]


def _fmt_pt(value: float, decimals: int = 1) -> str:
    """Formata número no padrão brasileiro (ponto para milhar, vírgula para decimal)."""
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def _apply_slide_chart_layout(
    fig: go.Figure,
    title: str,
    height: int = 400,
    top_margin: int = 84,
) -> None:
    """Padroniza layout escuro limpo para os gráficos da apresentação em slides."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor=COLOR_CARD,
        font=dict(color=COLOR_TEXT, family="sans-serif", size=12),
        title=dict(
            text=title,
            x=0.01,
            xanchor="left",
            y=0.96,
            yanchor="top",
            font=dict(size=14, color=COLOR_TEXT),
        ),
        height=height,
        margin=dict(l=20, r=28, t=top_margin, b=30),
    )


def build_slide1_progression_figure(summary: Dict[str, Any]) -> go.Figure:
    """Slide 1: Evolução do MAE fora da amostra do Baseline ao modelo com clima."""
    labels = [
        "Baseline Semanal<br>(1 atributo: lag_7)",
        "LightGBM Base<br>(28 atributos sem clima)",
        "LightGBM + Clima D+7<br>(45 atributos, erro 1,5°C)",
        "LightGBM + Clima ERA5<br>(teto clima realizado)",
    ]
    maes = [
        summary["baseline_mae"],
        summary["lgbm_base_mae"],
        summary["lgbm_clima_prev_mae"],
        summary["lgbm_clima_era5_mae"],
    ]
    mapes = [
        summary["baseline_mape"],
        summary["lgbm_base_mape"],
        summary["lgbm_clima_prev_mape"],
        summary["lgbm_clima_era5_mape"],
    ]
    gains = [
        0.0,
        summary["lgbm_base_gain_mae_pct"],
        summary["lgbm_clima_prev_gain_mae_pct"],
        summary["lgbm_clima_era5_gain_mae_pct"],
    ]
    colors = [COLOR_BASELINE, COLOR_REAL, COLOR_FORECAST, COLOR_TEMP]

    texts = []
    for mae, mape, gain in zip(maes, mapes, gains, strict=True):
        if gain == 0.0:
            texts.append(f"{_fmt_pt(mae, 0)} MWmed (MAPE {_fmt_pt(mape, 2)}%)")
        else:
            texts.append(f"{_fmt_pt(mae, 0)} MWmed (-{_fmt_pt(gain, 1)}%)")

    fig = go.Figure(
        data=[
            go.Bar(
                x=maes,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=texts,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>MAE: %{x:,.1f} MWmed<extra></extra>",
            )
        ]
    )
    max_mae = max(maes) if maes else 2500.0
    _apply_slide_chart_layout(
        fig,
        title="Redução Progressiva do MAE Out-of-Sample (1.049 dias | 12 Janelas)",
        height=390,
        top_margin=56,
    )
    fig.update_layout(
        margin=dict(l=20, r=36, t=56, b=30),
        xaxis=dict(
            title="MAE (MWmed) — menor é melhor",
            range=[0, max_mae * 1.62],
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat=",.0f",
            tickfont=dict(color=COLOR_MUTED),
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            showgrid=False,
            automargin=True,
            tickfont=dict(color=COLOR_TEXT, size=11),
        ),
    )
    return fig


def build_slide2_daytype_figure(day_breakdown: pd.DataFrame) -> go.Figure:
    """Slide 2: Compara MAE do Baseline vs LightGBM por categoria de dia do calendário."""
    df = day_breakdown.copy()
    y_labels = [
        f"{row['categoria']}<br>(n={int(row['n_dias'])} dias | -{_fmt_pt(row['ganho_pct'], 1)}%)"
        for _, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=y_labels,
            x=df["mae_baseline"],
            name="Baseline Semanal (t-7)",
            orientation="h",
            marker_color=COLOR_BASELINE,
            text=[f"{_fmt_pt(v, 0)} MWmed" for v in df["mae_baseline"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>MAE Baseline: %{x:,.1f} MWmed<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=y_labels,
            x=df["mae_lgbm"],
            name="LightGBM (D+7)",
            orientation="h",
            marker_color=COLOR_FORECAST,
            text=[f"{_fmt_pt(v, 0)} MWmed" for v in df["mae_lgbm"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>MAE LightGBM: %{x:,.1f} MWmed<extra></extra>",
        )
    )

    max_x = float(df["mae_baseline"].max()) if not df.empty else 5000.0
    _apply_slide_chart_layout(
        fig,
        title="MAE por Tipo de Dia: Onde o Baseline Quebra e o Modelo Corrige",
        height=410,
    )
    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=COLOR_MUTED)),
        xaxis=dict(
            title="MAE (MWmed)",
            range=[0, max_x * 1.24],
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat=",.0f",
            tickfont=dict(color=COLOR_MUTED),
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            showgrid=False,
            automargin=True,
            tickfont=dict(color=COLOR_TEXT, size=11),
        ),
    )
    return fig


def build_slide3_holiday_echo_figure(eval_enriched: pd.DataFrame) -> go.Figure:
    """Slide 3: Zoom na virada 2024/2025 evidenciando o Efeito Eco D+7 pós-feriado."""
    mask = (eval_enriched["data_referencia"] >= pd.Timestamp("2024-12-27")) & (
        eval_enriched["data_referencia"] <= pd.Timestamp("2025-01-13")
    )
    df = eval_enriched.loc[mask].sort_values("data_referencia").copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["data_referencia"],
            y=df["y_true"],
            mode="lines+markers",
            name="Carga Real (ONS)",
            line=dict(color=COLOR_REAL, width=2.5),
            marker=dict(size=6),
            hovertemplate="Data: %{x|%d/%m/%Y}<br>Carga Real: %{y:,.1f} MWmed<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["data_referencia"],
            y=df["y_pred_baseline"],
            mode="lines+markers",
            name="Baseline Semanal (t-7)",
            line=dict(color=COLOR_BASELINE, width=1.8, dash="dot"),
            marker=dict(size=5),
            hovertemplate="Data: %{x|%d/%m/%Y}<br>Baseline: %{y:,.1f} MWmed<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["data_referencia"],
            y=df["y_pred_lgbm"],
            mode="lines+markers",
            name="LightGBM (D+7)",
            line=dict(color=COLOR_FORECAST, width=2.3, dash="dash"),
            marker=dict(size=6),
            hovertemplate="Data: %{x|%d/%m/%Y}<br>LightGBM: %{y:,.1f} MWmed<extra></extra>",
        )
    )

    if not df.empty:
        row_jan1 = df[df["data_referencia"] == pd.Timestamp("2025-01-01")]
        if not row_jan1.empty:
            y_jan1 = float(row_jan1["y_true"].iloc[0])
            fig.add_annotation(
                x=pd.Timestamp("2025-01-01"),
                y=y_jan1,
                text="01/01 Feriado<br>(35.904 MWmed)",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=38,
                font=dict(size=10, color=COLOR_TEXT),
                bgcolor="#1e293b",
                bordercolor=COLOR_BORDER,
            )

        row_jan8 = df[df["data_referencia"] == pd.Timestamp("2025-01-08")]
        if not row_jan8.empty:
            y_base_jan8 = float(row_jan8["y_pred_baseline"].iloc[0])
            fig.add_annotation(
                x=pd.Timestamp("2025-01-08"),
                y=y_base_jan8,
                text="08/01 Eco D+7: Baseline<br>copia 01/01 (erro 11.210 MWmed)",
                showarrow=True,
                arrowhead=2,
                ax=45,
                ay=34,
                font=dict(size=10, color="#fca5a5"),
                bgcolor="#1e293b",
                bordercolor=COLOR_ERROR,
            )

    _apply_slide_chart_layout(
        fig,
        title="Caso Real 1: Ano Novo 2025 e Eco D+7 em 08/01/2025",
        height=390,
    )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=COLOR_MUTED)),
        xaxis=dict(title="", showgrid=False, tickformat="%d/%m", tickfont=dict(color=COLOR_MUTED)),
        yaxis=dict(
            title="Carga de Energia (MWmed)",
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat=",.0f",
            tickfont=dict(color=COLOR_MUTED),
        ),
    )
    return fig


def build_slide4_shock_damping_figure(eval_enriched: pd.DataFrame) -> go.Figure:
    """Slide 4: Zoom em março/2024 mostrando amortecimento de pico térmico e feriado."""
    mask = (eval_enriched["data_referencia"] >= pd.Timestamp("2024-03-15")) & (
        eval_enriched["data_referencia"] <= pd.Timestamp("2024-03-31")
    )
    df = eval_enriched.loc[mask].sort_values("data_referencia").copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["data_referencia"],
            y=df["y_true"],
            mode="lines+markers",
            name="Carga Real (ONS)",
            line=dict(color=COLOR_REAL, width=2.5),
            marker=dict(size=6),
            hovertemplate="Data: %{x|%d/%m/%Y}<br>Carga Real: %{y:,.1f} MWmed<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["data_referencia"],
            y=df["y_pred_baseline"],
            mode="lines+markers",
            name="Baseline Semanal (t-7)",
            line=dict(color=COLOR_BASELINE, width=1.8, dash="dot"),
            marker=dict(size=5),
            hovertemplate="Data: %{x|%d/%m/%Y}<br>Baseline: %{y:,.1f} MWmed<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["data_referencia"],
            y=df["y_pred_lgbm"],
            mode="lines+markers",
            name="LightGBM (D+7)",
            line=dict(color=COLOR_FORECAST, width=2.3, dash="dash"),
            marker=dict(size=6),
            hovertemplate="Data: %{x|%d/%m/%Y}<br>LightGBM: %{y:,.1f} MWmed<extra></extra>",
        )
    )

    if not df.empty:
        row_26 = df[df["data_referencia"] == pd.Timestamp("2024-03-26")]
        if not row_26.empty:
            fig.add_annotation(
                x=pd.Timestamp("2024-03-26"),
                y=float(row_26["y_pred_baseline"].iloc[0]),
                text="Baseline repete pico anterior<br>(erro 7.712 MWmed)",
                showarrow=True,
                arrowhead=2,
                ax=-25,
                ay=-38,
                font=dict(size=10, color="#fca5a5"),
                bgcolor="#1e293b",
                bordercolor=COLOR_ERROR,
            )
            fig.add_annotation(
                x=pd.Timestamp("2024-03-26"),
                y=float(row_26["y_pred_lgbm"].iloc[0]),
                text="LightGBM reverte à média<br>(erro 145 MWmed)",
                showarrow=True,
                arrowhead=2,
                ax=-55,
                ay=42,
                font=dict(size=10, color="#fdba74"),
                bgcolor="#1e293b",
                bordercolor=COLOR_FORECAST,
            )

    _apply_slide_chart_layout(
        fig,
        title="Caso Real 2: Reversão Térmica e Sexta-Feira Santa (Março/2024)",
        height=390,
    )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=COLOR_MUTED)),
        xaxis=dict(title="", showgrid=False, tickformat="%d/%m", tickfont=dict(color=COLOR_MUTED)),
        yaxis=dict(
            title="Carga de Energia (MWmed)",
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat=",.0f",
            tickfont=dict(color=COLOR_MUTED),
        ),
    )
    return fig


def build_slide5_feature_importance_figure(
    group_base: pd.DataFrame,
    group_clima: pd.DataFrame,
) -> go.Figure:
    """Slide 5: Compara participação (%) dos grupos de features com e sem clima."""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Modelo Base (28 Features sem Clima)",
            "Modelo + Clima D+7 (45 Features)",
        ),
        vertical_spacing=0.22,
    )

    gb = group_base.sort_values("gain_pct", ascending=True)
    fig.add_trace(
        go.Bar(
            x=gb["gain_pct"],
            y=gb["grupo"],
            orientation="h",
            marker_color=COLOR_REAL,
            text=[f"{_fmt_pt(v, 1)}%" for v in gb["gain_pct"]],
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
            hovertemplate="%{y}<br>Participação no Ganho: %{x:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    gc = group_clima.sort_values("gain_pct", ascending=True)
    clima_colors = [COLOR_TEMP if "Clima" in str(g) else COLOR_FORECAST for g in gc["grupo"]]
    fig.add_trace(
        go.Bar(
            x=gc["gain_pct"],
            y=gc["grupo"],
            orientation="h",
            marker_color=clima_colors,
            text=[f"{_fmt_pt(v, 1)}%" for v in gc["gain_pct"]],
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
            hovertemplate="%{y}<br>Participação no Ganho: %{x:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    _apply_slide_chart_layout(
        fig,
        title="Distribuição da Importância (Gain %) por Família de Atributos",
        height=460,
        top_margin=82,
    )
    x_max = max(float(gb["gain_pct"].max()), float(gc["gain_pct"].max())) * 1.24
    fig.update_layout(margin=dict(l=20, r=36, t=82, b=32))
    fig.update_annotations(font=dict(size=12, color=COLOR_MUTED))
    fig.update_xaxes(
        range=[0, x_max],
        showgrid=True,
        gridcolor=COLOR_GRID,
        tickfont=dict(color=COLOR_MUTED),
    )
    fig.update_xaxes(title_text="% do Ganho Total (Gain)", row=2, col=1)
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(color=COLOR_TEXT, size=10.5))
    return fig


def build_slide6_fold5_climate_figure(fold_comparison: pd.DataFrame) -> go.Figure:
    """Slide 6: MAE por Fold destacando o Fold 5 (Verão 2025) e o impacto do clima."""
    labels = [f"F{int(f)}" for f in fold_comparison["fold"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=fold_comparison["baseline_mae_mwmed"],
            name="Baseline Semanal (t-7)",
            marker_color=COLOR_BASELINE,
            hovertemplate="%{x}<br>Baseline: %{y:,.1f} MWmed<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=fold_comparison["lgbm_mae_mwmed"],
            name="LightGBM Sem Clima",
            marker_color=COLOR_REAL,
            hovertemplate="%{x}<br>LightGBM Sem Clima: %{y:,.1f} MWmed<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=fold_comparison["lgbm_clima_prev_15c_mae"],
            name="LightGBM + Clima (Prev. σ=1,5°C)",
            marker_color=COLOR_TEMP,
            hovertemplate="%{x}<br>LightGBM + Clima D+7: %{y:,.1f} MWmed<extra></extra>",
        )
    )

    max_y = float(
        fold_comparison[["baseline_mae_mwmed", "lgbm_mae_mwmed", "lgbm_clima_prev_15c_mae"]]
        .max()
        .max()
    )
    fig.add_vrect(
        x0=3.55,
        x1=4.45,
        fillcolor="rgba(239, 68, 68, 0.12)",
        line_width=1,
        line_dash="dot",
        line_color=COLOR_ERROR,
    )
    fig.add_annotation(
        x="F5",
        y=max_y * 1.18,
        text="Fold 5 (Verão 2025)<br>3.397 -> 2.389 MWmed (-29,7%)",
        showarrow=False,
        font=dict(size=10.5, color="#fca5a5"),
        bgcolor="rgba(20, 24, 32, 0.92)",
        bordercolor=COLOR_ERROR,
        borderwidth=1,
        borderpad=4,
    )

    _apply_slide_chart_layout(
        fig,
        title="Desempenho por Janela (12 Folds): Clima Reverte Derrota no Verão 2025",
        height=410,
        top_margin=84,
    )
    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(color=COLOR_MUTED)),
        xaxis=dict(
            title="Janela Walk-Forward (90 dias)", showgrid=False, tickfont=dict(color=COLOR_MUTED)
        ),
        yaxis=dict(
            title="MAE (MWmed)",
            range=[0, max_y * 1.34],
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat=",.0f",
            tickfont=dict(color=COLOR_MUTED),
        ),
    )
    return fig


def build_slide7_error_tail_figure(error_quantiles: pd.DataFrame) -> go.Figure:
    """Slide 7: Compressão dos quantis de erro absoluto (P50 a P99) do Baseline vs LightGBM."""
    df = error_quantiles.copy()
    x_labels = [
        f"{row['percentil']}<br>(-{_fmt_pt(row['reducao_pct'], 1)}%)" for _, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=df["erro_baseline_mwmed"],
            name="Baseline Semanal (t-7)",
            marker_color=COLOR_BASELINE,
            text=[_fmt_pt(v, 0) for v in df["erro_baseline_mwmed"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Erro Baseline: %{y:,.1f} MWmed<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=df["erro_lgbm_mwmed"],
            name="LightGBM (D+7)",
            marker_color=COLOR_FORECAST,
            text=[_fmt_pt(v, 0) for v in df["erro_lgbm_mwmed"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Erro LightGBM: %{y:,.1f} MWmed<extra></extra>",
        )
    )

    max_y = float(df["erro_baseline_mwmed"].max()) if not df.empty else 8500.0
    _apply_slide_chart_layout(
        fig,
        title="Compressão da Cauda de Erros: Quantis de Erro Absoluto (MWmed)",
        height=390,
    )
    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=COLOR_MUTED)),
        xaxis=dict(
            title="Percentil da Distribuição de Erros",
            showgrid=False,
            tickfont=dict(color=COLOR_TEXT),
        ),
        yaxis=dict(
            title="Erro Absoluto (MWmed)",
            range=[0, max_y * 1.18],
            showgrid=True,
            gridcolor=COLOR_GRID,
            tickformat=",.0f",
            tickfont=dict(color=COLOR_MUTED),
        ),
    )
    return fig


def _render_callout_box(title: str, body_html: str, accent_color: str = COLOR_FORECAST) -> None:
    """Renderiza bloco explicativo escuro com borda lateral de destaque."""
    st.markdown(
        f"""
        <div style="
            background-color: {COLOR_BG};
            border: 1px solid {COLOR_BORDER};
            border-left: 4px solid {accent_color};
            border-radius: 8px;
            padding: 0.78rem 0.95rem;
            margin-bottom: 0.65rem;
        ">
            <div style="font-size: 0.88rem; font-weight: 700;
                color: {COLOR_TEXT}; margin-bottom: 0.3rem;">
                {title}
            </div>
            <div style="font-size: 0.83rem; color: {COLOR_MUTED}; line-height: 1.48;">
                {body_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_slide_header(
    tag: str,
    title: str,
    subtitle_html: str,
) -> None:
    """Renderiza cabeçalho padronizado de um slide dentro do card escuro."""
    st.markdown(
        f"""
        <div style="border-bottom: 1px solid {COLOR_BORDER};
            padding-bottom: 0.65rem; margin-bottom: 0.9rem;">
            <div style="
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                font-size: 0.74rem;
                letter-spacing: 0.05em;
                color: {COLOR_FORECAST};
                font-weight: 600;
                margin-bottom: 0.25rem;
            ">{tag}</div>
            <div style="font-size: 1.32rem; font-weight: 700;
                color: {COLOR_TEXT}; line-height: 1.25;">
                {title}
            </div>
            <div style="font-size: 0.88rem; color: {COLOR_MUTED};
                margin-top: 0.3rem; line-height: 1.45;">
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_slide_footer(takeaway_html: str) -> None:
    """Renderiza barra de conclusão prática no rodapé do slide."""
    st.markdown(
        f"""
        <div style="
            background-color: rgba(31, 119, 180, 0.10);
            border: 1px solid rgba(31, 119, 180, 0.35);
            border-radius: 8px;
            padding: 0.6rem 0.9rem;
            margin-top: 0.55rem;
            font-size: 0.84rem;
            color: {COLOR_TEXT};
            line-height: 1.42;
        ">
            <strong style="color: {COLOR_FORECAST};">Onde a mudança faz diferença:</strong>
            {takeaway_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_single_slide(
    slide_idx: int,
    analytics: Dict[str, Any],
    term_fn: Callable[[str, str | None], str],
) -> None:
    """Renderiza o conteúdo completo de um slide específico (0 a 6)."""
    s = analytics["overall_summary"]
    day_bd = analytics["day_type_breakdown"]
    eval_enr = analytics["eval_enriched"]
    fi_base = analytics["feature_importance_base"]
    group_base = analytics["group_importance_base"]
    fi_clima = analytics["feature_importance_clima"]
    group_clima = analytics["group_importance_clima"]
    fold_comp = analytics["fold_comparison"]
    err_q = analytics["error_quantiles"]

    def t(label: str, key: str | None = None) -> str:
        return term_fn(label, key)

    with st.container(border=True):
        if slide_idx == 0:
            _render_slide_header(
                SLIDE_DEFINITIONS[0]["tag"],
                "Por que o LightGBM supera o Baseline em 11/12 janelas (e 12/12 com Clima)",
                (
                    f"Avaliação rigorosa em {t('Backtesting')} walk-forward "
                    f"({_fmt_pt(s['n_days'], 0)} dias entre {s['start_date']} e "
                    f"{s['end_date']}, 12 {t('Fold')}s de 90 dias, sem vazamento temporal)."
                ),
            )
            c_left, c_right = st.columns([1.0, 1.25], gap="large")
            with c_left:
                _render_callout_box(
                    "1. Limite estrutural do Baseline (t-7)",
                    (
                        f"O {t('Baseline Ingênuo Semanal')} copia cegamente um único ponto "
                        f'passado <span style="white-space:nowrap;">({t("t-7")})</span> com '
                        f"peso de 100%. Ele assume que a semana atual será idêntica à "
                        f"anterior, resultando em {t('MAE')} de "
                        f"<strong>{_fmt_pt(s['baseline_mae'], 1)} {t('MWmed')}</strong> e "
                        f"{t('MAPE')} de <strong>{_fmt_pt(s['baseline_mape'], 2)}%</strong>."
                    ),
                    accent_color=COLOR_BASELINE,
                )
                _render_callout_box(
                    "2. Salto 1: Engenharia Temporal + Calendário + Regime ONS (28 features)",
                    (
                        "Ao combinar múltiplos lags (&ge;7d), médias móveis, feriados nacionais "
                        f"e o regime {t('MMGD')}, o {t('LightGBM')} reduz o {t('MAE')} para "
                        f"<strong>{_fmt_pt(s['lgbm_base_mae'], 1)} MWmed</strong> "
                        f"(<strong>-{_fmt_pt(s['lgbm_base_gain_mae_pct'], 2)}%</strong>), "
                        f"vencendo <strong>{s['lgbm_base_folds_won']}</strong> janelas."
                    ),
                    accent_color=COLOR_REAL,
                )
                _render_callout_box(
                    "3. Salto 2: Inclusão de Temperatura Exógena D+7 (45 features)",
                    (
                        f"Com previsão meteorológica realista (&sigma;=1,5°C via "
                        f"{t('Open-Meteo ERA5')}), o {t('MAE')} cai para "
                        f"<strong>{_fmt_pt(s['lgbm_clima_prev_mae'], 1)} MWmed</strong> "
                        f"(<strong>-{_fmt_pt(s['lgbm_clima_prev_gain_mae_pct'], 2)}%</strong> "
                        f"sobre o Baseline), vencendo "
                        f"<strong>{s['lgbm_clima_prev_folds_won']}</strong> janelas."
                    ),
                    accent_color=COLOR_TEMP,
                )
            with c_right:
                st.plotly_chart(
                    build_slide1_progression_figure(s),
                    width="stretch",
                    key=f"slide_chart_{slide_idx}",
                )

            diff_total = s["baseline_mae"] - s["lgbm_clima_prev_mae"]
            _render_slide_footer(
                "Substituir uma regra fixa de 1 variável por aprendizado não linear sobre "
                "memória multi-semana, calendário de feriados e clima reduz o erro médio "
                f"diário em até <strong>{_fmt_pt(diff_total, 0)} MWmed</strong>."
            )

        elif slide_idx == 1:
            _render_slide_header(
                SLIDE_DEFINITIONS[1]["tag"],
                (
                    "Onde o ganho acontece: redução de erro chega a 57,2% "
                    "quando o lag_7 é distorcido"
                ),
                (
                    "Decomposição empírica dos 1.049 dias de backtesting mostra que a vantagem "
                    "do LightGBM se concentra exatamente nos regimes em que a repetição semanal "
                    "quebra."
                ),
            )
            c_left, c_right = st.columns([0.95, 1.3], gap="large")
            with c_left:
                eco_row = day_bd[day_bd["categoria"].str.startswith("Eco D+7")].iloc[0]
                hol_row = day_bd[day_bd["categoria"] == "Feriado Nacional"].iloc[0]
                reg_row = day_bd[day_bd["categoria"] == "Dia Útil Regular"].iloc[0]

                _render_callout_box(
                    "Maior ganho: Eco D+7 pós-feriado (-57,2%)",
                    (
                        f"Quando houve feriado há 7 dias ({int(eco_row['n_dias'])} dias), o "
                        f"Baseline erra <strong>{_fmt_pt(eco_row['mae_baseline'], 0)} "
                        "MWmed</strong> ao projetar o feriado para um dia útil normal. O LightGBM "
                        f"cai para <strong>{_fmt_pt(eco_row['mae_lgbm'], 0)} MWmed</strong> "
                        f"(economia de <strong>{_fmt_pt(eco_row['reducao_mwmed'], 0)} "
                        "MWmed/dia</strong>)."
                    ),
                    accent_color=COLOR_FORECAST,
                )
                _render_callout_box(
                    "Segundo maior ganho: Feriados Nacionais (-36,5%)",
                    (
                        f"Nos {int(hol_row['n_dias'])} feriados avaliados, o Baseline erra "
                        f"<strong>{_fmt_pt(hol_row['mae_baseline'], 0)} MWmed</strong> por "
                        "copiar um dia útil. O LightGBM antecipa a queda via "
                        "<code>is_holiday</code> e reduz para "
                        f"<strong>{_fmt_pt(hol_row['mae_lgbm'], 0)} MWmed</strong>."
                    ),
                    accent_color=COLOR_REAL,
                )
                _render_callout_box(
                    "Consistência também em dias regulares (-9,2% a -12,4%)",
                    (
                        f"Mesmo em dias úteis comuns ({int(reg_row['n_dias'])} dias) e fins de "
                        "semana, as médias móveis e a sazonalidade anual superam o ruído de um "
                        f"único dia (<strong>{_fmt_pt(reg_row['mae_baseline'], 0)}</strong> vs "
                        f"<strong>{_fmt_pt(reg_row['mae_lgbm'], 0)} MWmed</strong>)."
                    ),
                    accent_color=COLOR_TEMP,
                )
            with c_right:
                st.plotly_chart(
                    build_slide2_daytype_figure(day_bd),
                    width="stretch",
                    key=f"slide_chart_{slide_idx}",
                )

            _render_slide_footer(
                "Os ganhos não são um ajuste estatístico difuso: as maiores reduções ocorrem "
                "de forma cirúrgica nos <strong>feriados (+36,5%)</strong>, no "
                "<strong>eco D+7 (+57,2%)</strong> e nos <strong>dias ponte (+14,8%)</strong>."
            )

        elif slide_idx == 2:
            _render_slide_header(
                SLIDE_DEFINITIONS[2]["tag"],
                (
                    "Caso Real 1: Por que o Baseline erra duas vezes por feriado — "
                    "e como o LightGBM evita"
                ),
                (
                    "Inspeção da virada 2024/2025: o feriado de 01/01 derruba a carga real, "
                    "e o Baseline propaga esse vale artificial para a quarta-feira útil de "
                    "08/01/2025."
                ),
            )
            c_left, c_right = st.columns([0.95, 1.3], gap="large")
            with c_left:
                _render_callout_box(
                    "Falha dupla da regra y(t) = y(t-7)",
                    (
                        "<strong>1. No feriado (01/01/2025):</strong> A carga real caiu para "
                        "35.904 MWmed.<br><strong>2. Sete dias depois (08/01/2025):</strong> "
                        "Em uma quarta-feira útil normal de <strong>47.114 MWmed</strong>, o "
                        "Baseline repetiu os 35.904 MWmed de 01/01, errando "
                        "<strong>11.210 MWmed (23,8%)</strong> em um único dia."
                    ),
                    accent_color=COLOR_ERROR,
                )
                _render_callout_box(
                    "Como as features mudam a decisão em 08/01/2025",
                    (
                        "O LightGBM lê <code>is_holiday = 0</code>, <code>dayofweek = 2</code> "
                        "e cruza com <code>lag_14</code>, <code>lag_21</code>, "
                        "<code>lag_364</code> e <code>rolling_mean_28d_lag7</code>. Ao notar "
                        "que <code>lag_7</code> está muito abaixo da média de 28 dias, ancora "
                        "a previsão em <strong>42.773 MWmed</strong>."
                    ),
                    accent_color=COLOR_FORECAST,
                )
                _render_callout_box(
                    "Resultado prático comprovado",
                    (
                        "Apenas no dia 08/01/2025 o erro cai de <strong>11.210 MWmed</strong> "
                        "para <strong>4.342 MWmed</strong> (-6.869 MWmed / -61,3%). O mesmo "
                        "padrão se repete em 08/01/2024 (-6.333 MWmed) e após Carnaval, "
                        "Sexta-Feira Santa e 7 de Setembro."
                    ),
                    accent_color=COLOR_TEMP,
                )
            with c_right:
                st.plotly_chart(
                    build_slide3_holiday_echo_figure(eval_enr),
                    width="stretch",
                    key=f"slide_chart_{slide_idx}",
                )

            _render_slide_footer(
                "As variáveis de calendário (<code>is_holiday</code>, "
                "<code>is_holiday_eve</code>, <code>is_post_holiday</code>, "
                "<code>is_bridge_day</code>) combinadas a lags de 14–364 dias impedem que um "
                "feriado contamine a semana seguinte."
            )

        elif slide_idx == 3:
            _render_slide_header(
                SLIDE_DEFINITIONS[3]["tag"],
                (
                    "Caso Real 2: Quando a semana anterior foi atípica, "
                    "médias móveis e lag_364 estabilizam"
                ),
                (
                    "Episódio de março de 2024: onda de calor eleva a carga acima de 52.000 "
                    "MWmed e, na semana seguinte, o Baseline projeta um pico fantasma na queda "
                    "térmica e Sexta-Feira Santa."
                ),
            )
            c_left, c_right = st.columns([0.95, 1.3], gap="large")
            with c_left:
                _render_callout_box(
                    "O choque de 18 a 21/03/2024 e o efeito fantasma",
                    (
                        "Entre 18 e 21/03, máximas acima de 30°C elevaram a carga para "
                        "<strong>52.295–52.934 MWmed</strong>. Na semana seguinte (25 a 29/03), "
                        "a temperatura caiu ~5°C e a carga voltou a <strong>44.000–45.200 "
                        "MWmed</strong> (e 39.041 MWmed no feriado de 29/03)."
                    ),
                    accent_color=COLOR_BASELINE,
                )
                _render_callout_box(
                    "Por que o Baseline errou 5 dias seguidos (>7.700 MWmed/dia)",
                    (
                        "Sem memória além de <code>t-7</code>, o Baseline copiou o pico da "
                        "onda de calor para toda a semana de 25 a 29/03, gerando erros "
                        "consecutivos de <strong>7.946</strong>, <strong>7.712</strong>, "
                        "<strong>7.974</strong>, <strong>8.336</strong> e "
                        "<strong>9.715 MWmed</strong>."
                    ),
                    accent_color=COLOR_ERROR,
                )
                _render_callout_box(
                    "Amortecimento do LightGBM (erros de 89 e 145 MWmed)",
                    (
                        "Os atributos <code>ratio_lag7_rolling28</code>, "
                        "<code>diff_lag7_lag14</code> e <code>lag_364</code> sinalizam que "
                        "<code>lag_7</code> estava esticado acima do outono. O LightGBM previu "
                        "<strong>44.650 MWmed</strong> em 25/03 (erro 89 MWmed) e "
                        "<strong>45.077 MWmed</strong> em 26/03 (erro 145 MWmed)."
                    ),
                    accent_color=COLOR_TEMP,
                )
            with c_right:
                st.plotly_chart(
                    build_slide4_shock_damping_figure(eval_enr),
                    width="stretch",
                    key=f"slide_chart_{slide_idx}",
                )

            _render_slide_footer(
                "Os indicadores relativos (<code>diff_lag7_lag14</code> e "
                "<code>ratio_lag7_rolling28</code>) funcionam como filtro de reversão à média: "
                "evitam que choques pontuais da semana passada sejam repetidos no horizonte D+7."
            )

        elif slide_idx == 4:
            _render_slide_header(
                SLIDE_DEFINITIONS[4]["tag"],
                (
                    "Anatomia das Features: Como o modelo distribui peso entre memória, "
                    "calendário e clima"
                ),
                (
                    "Comparação direta da importância (Gain %) no LightGBM sem clima (28 "
                    "atributos) e no modelo enriquecido com temperatura ponderada do SE (45 "
                    "atributos)."
                ),
            )
            c_left, c_right = st.columns([0.95, 1.3], gap="large")
            with c_left:
                lag7_base = float(fi_base.loc[fi_base["feature"] == "lag_7", "gain_pct"].iloc[0])
                lag7_clima = float(
                    fi_clima.loc[fi_clima["feature"] == "lag_7", "gain_pct"].iloc[0]
                )
                top_clima = fi_clima[fi_clima["grupo"].str.contains("Clima")].head(3)
                top_clima_txt = ", ".join(
                    f"<code>{r['feature']}</code> ({_fmt_pt(r['gain_pct'], 1)}%)"
                    for _, r in top_clima.iterrows()
                )

                _render_callout_box(
                    "1. Desconcentração do lag_7 (100% -> 30,7% -> 16,2%)",
                    (
                        "No Baseline, <code>lag_7</code> responde por 100% da previsão. No "
                        f"LightGBM Base, cai para <strong>{_fmt_pt(lag7_base, 1)}%</strong>; "
                        f"com clima, cai para <strong>{_fmt_pt(lag7_clima, 1)}%</strong>, pois "
                        "<code>lag_14</code>, <code>lag_21</code> e <code>lag_364</code> "
                        "dividem a ancoragem."
                    ),
                    accent_color=COLOR_REAL,
                )
                _render_callout_box(
                    "2. Regime ONS (MMGD) evita viés histórico de 2018–2022",
                    (
                        "A variável <code>regime_medicao_code</code> separa as três fases "
                        f"metodológicas do {t('ONS')}, permitindo treinar desde 2018 "
                        f"mantendo calibragem ao nível pós-inclusão da {t('MMGD')} (abril/2023)."
                    ),
                    accent_color=COLOR_FORECAST,
                )
                _render_callout_box(
                    "3. Clima entra no Top 6 do modelo (anomalia e delta térmico)",
                    (
                        "Ao adicionar temperatura, as principais variáveis climáticas são "
                        f"{top_clima_txt}, capturando o efeito não linear do ar-condicionado."
                    ),
                    accent_color=COLOR_TEMP,
                )
            with c_right:
                st.plotly_chart(
                    build_slide5_feature_importance_figure(group_base, group_clima),
                    width="stretch",
                    key=f"slide_chart_{slide_idx}",
                )

            _render_slide_footer(
                "Feriados e Regime ONS possuem menos divisões na árvore porque são eventos "
                "raros ou estruturais, mas concentram altíssimo impacto exatamente nos dias "
                "críticos."
            )

        elif slide_idx == 5:
            _render_slide_header(
                SLIDE_DEFINITIONS[5]["tag"],
                (
                    "Ponto Cego Sem Clima (Fold 5) e por que a Temperatura Exógena "
                    "dobra o ganho"
                ),
                (
                    "Diagnóstico transparente da única janela em que o Baseline superou o "
                    "LightGBM sem clima (Verão 2025: 26/12/2024 a 25/03/2025) e como a "
                    "previsão térmica D+7 resolveu a falha."
                ),
            )
            c_left, c_right = st.columns([0.95, 1.3], gap="large")
            with c_left:
                _render_callout_box(
                    "Por que o LightGBM Sem Clima perdeu no Fold 5?",
                    (
                        "Nos outros 11 folds, o LightGBM vence com ganho médio de "
                        f"<strong>+{_fmt_pt(s['other_folds_gain_pct'], 2)}%</strong>. No Fold 5 "
                        f"(Verão 2025, média de <strong>{_fmt_pt(s['fold5_mean_temp'], 1)}°C"
                        f"</strong> e máxima de <strong>{_fmt_pt(s['fold5_max_temp'], 1)}°C"
                        "</strong>), ondas de calor persistentes elevaram a carga. Sem "
                        "termômetro, o modelo puxou para a média sazonal (viés de "
                        f"<strong>+{_fmt_pt(s['fold5_bias_lgbm'], 0)} MWmed</strong> abaixo do "
                        "real), enquanto o <code>lag_7</code> carregava o calor recente."
                    ),
                    accent_color=COLOR_ERROR,
                )
                _render_callout_box(
                    "Lags climáticos (>=7d) não bastam; previsão D+7 faz a diferença",
                    (
                        "Testamos 3 cenários sem vazamento: usar só temperatura passada (&ge;7d) "
                        "quase não altera o Fold 5 (3.525 MWmed). Já incluir a "
                        "<strong>previsão térmica D+7 (&sigma;=1,5°C)</strong> reduz o MAE do "
                        f"Fold 5 de <strong>{_fmt_pt(s['fold5_lgbm_base_mae'], 0)}</strong> "
                        f"para <strong>{_fmt_pt(s['fold5_lgbm_clima_prev_mae'], 0)} MWmed "
                        "(-29,7%)</strong>."
                    ),
                    accent_color=COLOR_TEMP,
                )
                _render_callout_box(
                    "Placar final com Clima D+7: 12 de 12 janelas vencidas",
                    (
                        "O ganho global sobre o Baseline salta de <strong>+11,38%</strong> "
                        f"para <strong>+{_fmt_pt(s['lgbm_clima_prev_gain_mae_pct'], 2)}%</strong> "
                        f"(chegando a <strong>+{_fmt_pt(s['lgbm_clima_era5_gain_mae_pct'], 2)}%"
                        "</strong> com previsão meteorológica mais precisa)."
                    ),
                    accent_color=COLOR_FORECAST,
                )
            with c_right:
                st.plotly_chart(
                    build_slide6_fold5_climate_figure(fold_comp),
                    width="stretch",
                    key=f"slide_chart_{slide_idx}",
                )

            _render_slide_footer(
                "A temperatura futura prevista (<code>temp_anomalia_7d</code>, "
                "<code>temp_delta_lag7</code>, <code>cdd_target</code>) explica a demanda "
                "térmica de ar-condicionado que nenhuma memória histórica de carga antecipa."
            )

        elif slide_idx == 6:
            _render_slide_header(
                SLIDE_DEFINITIONS[6]["tag"],
                "Redução de Erros Extremos e Síntese Executiva das Mudanças",
                (
                    "Impacto na cauda da distribuição de erros ao longo de 1.049 dias e "
                    "roteiro objetivo para evolução em produção."
                ),
            )
            c_left, c_right = st.columns([1.0, 1.25], gap="large")
            with c_left:
                _render_callout_box(
                    (
                        f"Corte de -{_fmt_pt(s['extreme_gain_pct'], 1)}% nos dias de erro "
                        "extremo (>5.000 MWmed)"
                    ),
                    (
                        f"Nos <strong>{s['extreme_days_count']} dias</strong> em que o Baseline "
                        f"errou acima de 5.000 MWmed (MAE médio de "
                        f"<strong>{_fmt_pt(s['extreme_base_mae'], 0)} MWmed</strong>), o "
                        f"LightGBM reduziu para <strong>{_fmt_pt(s['extreme_lgbm_mae'], 0)} "
                        "MWmed</strong>. Dias dentro de &plusmn;2.000 MWmed subiram de "
                        f"<strong>{_fmt_pt(s['days_within_2k_base_pct'], 1)}%</strong> para "
                        f"<strong>{_fmt_pt(s['days_within_2k_lgbm_pct'], 1)}%</strong>."
                    ),
                    accent_color=COLOR_FORECAST,
                )
                _render_callout_box(
                    "Resumo: Alavanca -> Problema Resolvido -> Ganho",
                    (
                        "<strong>1. Lags &ge;7d + Médias Móveis:</strong> amortece ruído "
                        "semanal (-41,1% em extremos).<br>"
                        "<strong>2. Calendário BR + Pontes:</strong> corrige feriados "
                        "(-36,5%) e eco D+7 (-57,2%).<br>"
                        "<strong>3. Regime ONS (MMGD):</strong> calibra mudança estrutural "
                        "pós-abr/2023.<br>"
                        "<strong>4. Clima D+7 (ERA5/CDD):</strong> resolve ondas de calor "
                        "(Fold 5) e eleva ganho a +24,61%."
                    ),
                    accent_color=COLOR_REAL,
                )
                _render_callout_box(
                    "Próximos passos recomendados para operação",
                    (
                        "Integrar previsões meteorológicas reais D+7 (Open-Meteo Forecast / "
                        "CPTEC) em produção, expandir a modelagem global para os 4 subsistemas "
                        "(SE, S, NE, N) e comparar com baseline estatístico (AutoETS/AutoARIMA)."
                    ),
                    accent_color=COLOR_TEMP,
                )
            with c_right:
                st.plotly_chart(
                    build_slide7_error_tail_figure(err_q),
                    width="stretch",
                    key=f"slide_chart_{slide_idx}",
                )

            _render_slide_footer(
                "O ganho do modelo não vem de complexidade opaca, mas de quatro correções "
                "interpretáveis: <strong>filtragem multi-semana</strong>, "
                "<strong>calendário de feriados</strong>, <strong>regime MMGD</strong> e "
                "<strong>sensibilidade climática D+7</strong>."
            )


def render_presentation_tab(
    analytics: Dict[str, Any],
    term_fn: Callable[[str, str | None], str],
) -> None:
    """Renderiza a apresentação interativa em slides dentro da aba dedicada do Streamlit."""
    total_slides = len(SLIDE_DEFINITIONS)

    if "presentation_slide_idx" not in st.session_state:
        st.session_state["presentation_slide_idx"] = 0
    if st.session_state["presentation_slide_idx"] < 0:
        st.session_state["presentation_slide_idx"] = 0
    if st.session_state["presentation_slide_idx"] >= total_slides:
        st.session_state["presentation_slide_idx"] = total_slides - 1

    def _go_prev() -> None:
        st.session_state["presentation_slide_idx"] = max(
            0, int(st.session_state["presentation_slide_idx"]) - 1
        )

    def _go_next() -> None:
        st.session_state["presentation_slide_idx"] = min(
            total_slides - 1, int(st.session_state["presentation_slide_idx"]) + 1
        )

    current_idx = int(st.session_state["presentation_slide_idx"])

    with st.container(border=True):
        c_prev, c_select, c_next, c_all = st.columns([1.15, 3.3, 1.15, 1.6])
        with c_prev:
            st.button(
                "Slide Anterior",
                on_click=_go_prev,
                disabled=(current_idx == 0),
                width="stretch",
                key="btn_slide_prev",
            )
        with c_select:
            st.selectbox(
                "Selecionar Slide",
                options=list(range(total_slides)),
                format_func=lambda i: (
                    f"Slide {i + 1}/{total_slides}: {SLIDE_DEFINITIONS[i]['nav_title']}"
                ),
                key="presentation_slide_idx",
                label_visibility="collapsed",
            )
        with c_next:
            st.button(
                "Próximo Slide",
                on_click=_go_next,
                disabled=(current_idx == total_slides - 1),
                width="stretch",
                key="btn_slide_next",
            )
        with c_all:
            show_all = st.checkbox(
                "Exibir todos os slides",
                value=False,
                key="chk_show_all_slides",
                help=(
                    "Alterna entre navegação slide a slide (modo apresentação) e visualização "
                    "contínua de todos os 7 slides."
                ),
            )

        current_idx = int(st.session_state["presentation_slide_idx"])
        progress_pct = ((current_idx + 1) / total_slides) * 100.0 if not show_all else 100.0
        status_label = (
            "Todos os 7 slides visíveis"
            if show_all
            else f"Slide {current_idx + 1} de {total_slides}"
        )
        st.markdown(
            f"""
            <div style="margin-top: 0.35rem;">
                <div style="display:flex;justify-content:space-between;
                    font-size:0.75rem;color:{COLOR_MUTED};margin-bottom:0.25rem;">
                    <span>Roteiro Explicativo: Por que o Modelo Vence o Baseline</span>
                    <span>{status_label}</span>
                </div>
                <div style="width:100%;height:4px;background:{COLOR_BORDER};
                    border-radius:999px;overflow:hidden;">
                    <div style="width:{progress_pct:.1f}%;height:100%;
                        background:{COLOR_FORECAST};transition:width 0.2s ease;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if show_all:
        for idx in range(total_slides):
            _render_single_slide(idx, analytics, term_fn)
    else:
        _render_single_slide(current_idx, analytics, term_fn)


@st.cache_data(show_spinner="Carregando evidências e slides analíticos...")
def load_slide_analytics_bundle() -> Dict[str, Any]:
    """Calcula e armazena em cache as evidências explicativas da apresentação em slides."""
    data_mod = importlib.import_module("src.app.08_dashboard_data")
    df_hist = data_mod.load_historical_series()
    bt_preds, bt_folds, clima_comp, clima_folds = data_mod.load_backtest_artifacts()
    return data_mod.compute_slide_analytics(df_hist, bt_preds, bt_folds, clima_comp, clima_folds)


def main() -> None:
    """Ponto de entrada standalone para a apresentação executiva em slides."""
    app_mod = importlib.import_module("src.app.08_dashboard_app")
    apply_dark_style = app_mod._apply_dark_style
    term = app_mod._term

    st.set_page_config(
        page_title="Apresentação Executiva: LightGBM vs Baseline - Subsistema SE (ONS)",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_dark_style()

    header_html = (
        '<div class="dashboard-header">'
        "<div>"
        '<h1 class="dashboard-title" style="font-size: 1.75rem; margin: 0;">'
        f"Apresentação Executiva: {term('LightGBM')} vs Baseline - "
        f'<span style="white-space:nowrap;">Subsistema {term("SE")} ({term("ONS")})</span>'
        "</h1>"
        '<p class="dashboard-subtitle">'
        f"Diagnóstico explicativo dos ganhos fora da amostra em {term('Backtesting')} "
        f"walk-forward (1.049 dias | 12 {term('Fold')}s), detalhando onde o "
        f"{term('Baseline Ingênuo Semanal')} falha e como cada mudança nas variáveis "
        f"corrige os erros em {term('MWmed')}."
        "</p>"
        "</div>"
        '<div class="dashboard-source">'
        f"Fonte: {term('ONS')} (2018-2026) | {term('Open-Meteo ERA5')} | {term('DuckDB Gold')}"
        "</div>"
        "</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

    slide_analytics = load_slide_analytics_bundle()
    render_presentation_tab(slide_analytics, term)


if __name__ == "__main__":
    main()
