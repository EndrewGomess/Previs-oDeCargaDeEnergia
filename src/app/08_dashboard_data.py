#!/usr/bin/env python3
"""08_dashboard_data.py

Camada de preparação de dados e inferência operacional (D+1 a D+7) para o Dashboard Streamlit.
Conecta-se diretamente ao warehouse DuckDB (camada Gold enriquecida com clima) e aos
artefatos reais de backtesting walk-forward gerados nas etapas 06 e 07.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

feat_mod = importlib.import_module("src.features.05_feature_engineering")
weather_feat_mod = importlib.import_module("src.features.07_feature_engineering_clima")
train_mod = importlib.import_module("src.models.05_train_lightgbm")

FEATURE_COLUMNS = feat_mod.FEATURE_COLUMNS
build_features = feat_mod.build_features
WEATHER_LAG_FEATURES = weather_feat_mod.WEATHER_LAG_FEATURES
WEATHER_FORECAST_FEATURES = weather_feat_mod.WEATHER_FORECAST_FEATURES
build_weather_features = weather_feat_mod.build_weather_features
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
                temp_min_se,
                temp_aparente_se,
                temp_max_sp
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


FEATURE_METADATA: Dict[str, Tuple[str, str]] = {
    "lag_7": ("Carga de 7 dias atrás (t-7)", "Lags Temporais (>=7d)"),
    "lag_8": ("Carga de 8 dias atrás (t-8)", "Lags Temporais (>=7d)"),
    "lag_9": ("Carga de 9 dias atrás (t-9)", "Lags Temporais (>=7d)"),
    "lag_14": ("Carga de 14 dias atrás (t-14)", "Lags Temporais (>=7d)"),
    "lag_21": ("Carga de 21 dias atrás (t-21)", "Lags Temporais (>=7d)"),
    "lag_28": ("Carga de 28 dias atrás (t-28)", "Lags Temporais (>=7d)"),
    "lag_364": ("Carga de 52 semanas atrás (t-364)", "Lags Temporais (>=7d)"),
    "rolling_mean_7d_lag7": ("Média móvel 7d (até t-7)", "Tendência e Volatilidade Móvel"),
    "rolling_std_7d_lag7": ("Desvio padrão 7d (até t-7)", "Tendência e Volatilidade Móvel"),
    "rolling_mean_28d_lag7": ("Média móvel 28d (até t-7)", "Tendência e Volatilidade Móvel"),
    "rolling_std_28d_lag7": ("Desvio padrão 28d (até t-7)", "Tendência e Volatilidade Móvel"),
    "diff_lag7_lag14": ("Variação semanal (t-7 vs t-14)", "Tendência e Volatilidade Móvel"),
    "ratio_lag7_rolling28": ("Razão t-7 / Média 28d", "Tendência e Volatilidade Móvel"),
    "dayofweek": ("Dia da semana (0-6)", "Calendário e Sazonalidade"),
    "is_weekend": ("Indicador de fim de semana", "Calendário e Sazonalidade"),
    "day": ("Dia do mês", "Calendário e Sazonalidade"),
    "month": ("Mês do ano", "Calendário e Sazonalidade"),
    "dayofyear": ("Dia do ano (1-366)", "Calendário e Sazonalidade"),
    "weekofyear": ("Semana do ano", "Calendário e Sazonalidade"),
    "sin_dayofweek": ("Ciclo semanal (seno)", "Calendário e Sazonalidade"),
    "cos_dayofweek": ("Ciclo semanal (cosseno)", "Calendário e Sazonalidade"),
    "sin_dayofyear": ("Sazonalidade anual (seno)", "Calendário e Sazonalidade"),
    "cos_dayofyear": ("Sazonalidade anual (cosseno)", "Calendário e Sazonalidade"),
    "is_holiday": ("Feriado nacional (BR)", "Feriados e Pontes"),
    "is_holiday_eve": ("Véspera de feriado", "Feriados e Pontes"),
    "is_post_holiday": ("Dia pós-feriado", "Feriados e Pontes"),
    "is_bridge_day": ("Dia ponte (emenda de feriado)", "Feriados e Pontes"),
    "regime_medicao_code": ("Regime ONS (Despachada / MMGD)", "Regime Metodológico ONS"),
    "temp_media_lag7": ("Temp. média SE (t-7)", "Clima Exógeno (Lags >=7d)"),
    "temp_max_lag7": ("Temp. máxima SE (t-7)", "Clima Exógeno (Lags >=7d)"),
    "temp_aparente_lag7": ("Sensação térmica SE (t-7)", "Clima Exógeno (Lags >=7d)"),
    "temp_max_sp_lag7": ("Temp. máxima SP (t-7)", "Clima Exógeno (Lags >=7d)"),
    "temp_rolling_mean_7d_lag7": ("Temp. média 7d (até t-7)", "Clima Exógeno (Lags >=7d)"),
    "temp_rolling_max_7d_lag7": ("Temp. máxima 7d (até t-7)", "Clima Exógeno (Lags >=7d)"),
    "cdd_lag7": ("Graus-dia resfriamento CDD (t-7)", "Clima Exógeno (Lags >=7d)"),
    "temp_media_target": ("Temp. média prevista (D+7)", "Clima Exógeno (Previsão D+7)"),
    "temp_max_target": ("Temp. máxima prevista (D+7)", "Clima Exógeno (Previsão D+7)"),
    "temp_min_target": ("Temp. mínima prevista (D+7)", "Clima Exógeno (Previsão D+7)"),
    "temp_aparente_target": ("Sensação térmica prevista (D+7)", "Clima Exógeno (Previsão D+7)"),
    "temp_max_sp_target": ("Temp. máxima SP prevista (D+7)", "Clima Exógeno (Previsão D+7)"),
    "temp_amplitude_target": ("Amplitude térmica prevista (D+7)", "Clima Exógeno (Previsão D+7)"),
    "cdd_target": ("CDD médio previsto >22°C (D+7)", "Clima Exógeno (Previsão D+7)"),
    "cdd_max_target": ("CDD máximo previsto >26°C (D+7)", "Clima Exógeno (Previsão D+7)"),
    "temp_anomalia_7d": ("Anomalia térmica vs média 7d", "Clima Exógeno (Previsão D+7)"),
    "temp_delta_lag7": ("Variação térmica (t vs t-7)", "Clima Exógeno (Previsão D+7)"),
}


def _compute_feature_importance_table(
    df_feat: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Treina LightGBM com split temporal e extrai importância por atributo e por grupo."""
    df_clean = df_feat.dropna(subset=feature_cols + ["carga_energia_mwmed"]).copy()
    last_date = df_clean["data_referencia"].max()
    val_cutoff = last_date - pd.Timedelta(days=90)

    train_df = df_clean[df_clean["data_referencia"] < val_cutoff]
    val_df = df_clean[df_clean["data_referencia"] >= val_cutoff]

    model = lgb.LGBMRegressor(**DEFAULT_LGBM_PARAMS)
    model.fit(
        train_df[feature_cols],
        train_df["carga_energia_mwmed"],
        eval_X=val_df[feature_cols],
        eval_y=val_df["carga_energia_mwmed"],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    gains = model.booster_.feature_importance(importance_type="gain").astype("float64")
    splits = model.booster_.feature_importance(importance_type="split").astype("int64")
    total_gain = float(gains.sum()) if float(gains.sum()) > 0 else 1.0

    rows = []
    for col, g, s in zip(feature_cols, gains, splits, strict=True):
        label_pt, grupo = FEATURE_METADATA.get(col, (col, "Outros"))
        rows.append(
            {
                "feature": col,
                "label_pt": label_pt,
                "grupo": grupo,
                "importance_gain": float(g),
                "gain_pct": float(g / total_gain * 100.0),
                "splits": int(s),
            }
        )

    fi_df = (
        pd.DataFrame(rows)
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )
    group_df = (
        fi_df.groupby("grupo", as_index=False)
        .agg(
            n_features=("feature", "count"),
            gain_pct=("gain_pct", "sum"),
            importance_gain=("importance_gain", "sum"),
        )
        .sort_values("gain_pct", ascending=False)
        .reset_index(drop=True)
    )
    return fi_df, group_df


def compute_slide_analytics(
    df_hist: pd.DataFrame,
    bt_preds: pd.DataFrame,
    bt_folds: pd.DataFrame,
    clima_comp: pd.DataFrame,
    clima_folds: pd.DataFrame,
) -> Dict[str, Any]:
    """Calcula evidências empíricas explicativas para os slides de comparação vs Baseline."""
    df_base_feat = build_features(df_hist)
    feat_cols_merge = [
        "data_referencia",
        "dayofweek",
        "is_weekend",
        "is_holiday",
        "is_holiday_eve",
        "is_post_holiday",
        "is_bridge_day",
        "regime_medicao_ons",
        "temp_media_se",
        "temp_max_se",
    ]
    eval_enriched = bt_preds.merge(df_base_feat[feat_cols_merge], on="data_referencia", how="left")
    eval_enriched["err_base"] = (
        eval_enriched["y_true"] - eval_enriched["y_pred_baseline"]
    ).abs()
    eval_enriched["err_lgbm"] = (eval_enriched["y_true"] - eval_enriched["y_pred_lgbm"]).abs()
    eval_enriched["res_base"] = eval_enriched["y_true"] - eval_enriched["y_pred_baseline"]
    eval_enriched["res_lgbm"] = eval_enriched["y_true"] - eval_enriched["y_pred_lgbm"]
    eval_enriched["diff_err"] = eval_enriched["err_base"] - eval_enriched["err_lgbm"]

    # Identifica dias em que exatamente 7 dias antes houve feriado (contaminação do lag_7)
    holiday_dates = set(
        df_base_feat.loc[df_base_feat["is_holiday"] == 1, "data_referencia"]
    )
    eval_enriched["is_lag7_holiday"] = (
        eval_enriched["data_referencia"] - pd.Timedelta(days=7)
    ).isin(holiday_dates)

    # Quebra explicativa por categoria de dia
    categories = [
        (
            "Eco D+7 de Feriado (t-7 foi feriado)",
            eval_enriched["is_lag7_holiday"] & (eval_enriched["is_holiday"] == 0),
        ),
        ("Feriado Nacional", eval_enriched["is_holiday"] == 1),
        (
            "Pós-Feriado / Dia Ponte",
            (eval_enriched["is_post_holiday"] == 1) | (eval_enriched["is_bridge_day"] == 1),
        ),
        ("Véspera de Feriado", eval_enriched["is_holiday_eve"] == 1),
        (
            "Fim de Semana Regular",
            (eval_enriched["is_weekend"] == 1)
            & (eval_enriched["is_holiday"] == 0)
            & (~eval_enriched["is_lag7_holiday"]),
        ),
        (
            "Dia Útil Regular",
            (eval_enriched["is_weekend"] == 0)
            & (eval_enriched["is_holiday"] == 0)
            & (eval_enriched["is_post_holiday"] == 0)
            & (eval_enriched["is_bridge_day"] == 0)
            & (~eval_enriched["is_lag7_holiday"]),
        ),
    ]
    breakdown_rows = []
    for label, mask in categories:
        sub = eval_enriched.loc[mask]
        if sub.empty:
            continue
        mae_b = float(sub["err_base"].mean())
        mae_l = float(sub["err_lgbm"].mean())
        red = mae_b - mae_l
        gain_pct = (red / mae_b * 100.0) if mae_b > 0 else 0.0
        breakdown_rows.append(
            {
                "categoria": label,
                "n_dias": int(len(sub)),
                "mae_baseline": mae_b,
                "mae_lgbm": mae_l,
                "reducao_mwmed": red,
                "ganho_pct": gain_pct,
            }
        )
    day_type_breakdown = (
        pd.DataFrame(breakdown_rows)
        .sort_values("ganho_pct", ascending=False)
        .reset_index(drop=True)
    )

    # Quantis de erro absoluto e redução de erros extremos
    quantile_rows = []
    for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
        qb = float(eval_enriched["err_base"].quantile(q))
        ql = float(eval_enriched["err_lgbm"].quantile(q))
        quantile_rows.append(
            {
                "percentil": f"P{int(q * 100)}",
                "erro_baseline_mwmed": qb,
                "erro_lgbm_mwmed": ql,
                "reducao_mwmed": qb - ql,
                "reducao_pct": ((qb - ql) / qb * 100.0) if qb > 0 else 0.0,
            }
        )
    error_quantiles = pd.DataFrame(quantile_rows)

    # Importância de atributos (modelo base vs modelo climático)
    fi_base, group_base = _compute_feature_importance_table(df_base_feat, FEATURE_COLUMNS)

    all_weather_cols = FEATURE_COLUMNS + WEATHER_LAG_FEATURES + WEATHER_FORECAST_FEATURES
    df_weather_feat = build_weather_features(df_hist, forecast_noise_std=0.0)
    fi_clima, group_clima = _compute_feature_importance_table(df_weather_feat, all_weather_cols)

    # Consolidação dos 12 folds (Base + Clima)
    fold_comparison = bt_folds[
        ["fold", "test_start", "test_end", "baseline_mae_mwmed", "lgbm_mae_mwmed", "mae_gain_pct"]
    ].merge(
        clima_folds[
            [
                "fold",
                "lgbm_clima_lag7_mae",
                "lgbm_clima_prev_15c_mae",
                "lgbm_clima_era5_mae",
            ]
        ],
        on="fold",
        how="left",
    )

    # Resumo consolidado extraído dos artefatos reais
    clima_by_model = clima_comp.set_index("modelo")
    base_row = clima_by_model.loc["Baseline Ingênuo (Semanal)"]
    lgbm_ref_row = clima_by_model.loc["LightGBM Sem Clima (Ref.)"]
    lgbm_prev_row = clima_by_model.loc["LightGBM + Clima (Prev. D+7 σ=1.5°C)"]
    lgbm_era5_row = clima_by_model.loc["LightGBM + Clima (ERA5 Realizado)"]

    extreme_mask = eval_enriched["err_base"] > 5000.0
    extreme_sub = eval_enriched.loc[extreme_mask]
    ext_base_mae = float(extreme_sub["err_base"].mean()) if not extreme_sub.empty else 0.0
    ext_lgbm_mae = float(extreme_sub["err_lgbm"].mean()) if not extreme_sub.empty else 0.0

    fold5_eval = eval_enriched[eval_enriched["fold"] == 5]
    fold5_row = fold_comparison[fold_comparison["fold"] == 5].iloc[0]
    other_eval = eval_enriched[eval_enriched["fold"] != 5]

    overall_summary: Dict[str, Any] = {
        "n_days": int(len(eval_enriched)),
        "n_folds": int(len(fold_comparison)),
        "start_date": eval_enriched["data_referencia"].min().strftime("%d/%m/%Y"),
        "end_date": eval_enriched["data_referencia"].max().strftime("%d/%m/%Y"),
        "baseline_mae": float(base_row["mae_mwmed"]),
        "baseline_mape": float(base_row["mape_pct"]),
        "baseline_rmse": float(base_row["rmse_mwmed"]),
        "lgbm_base_mae": float(lgbm_ref_row["mae_mwmed"]),
        "lgbm_base_mape": float(lgbm_ref_row["mape_pct"]),
        "lgbm_base_rmse": float(lgbm_ref_row["rmse_mwmed"]),
        "lgbm_base_gain_mae_pct": float(lgbm_ref_row["ganho_mae_vs_base_pct"]),
        "lgbm_base_gain_mape_pct": float(lgbm_ref_row["ganho_mape_vs_base_pct"]),
        "lgbm_base_folds_won": str(lgbm_ref_row["folds_vencidos"]),
        "lgbm_clima_prev_mae": float(lgbm_prev_row["mae_mwmed"]),
        "lgbm_clima_prev_mape": float(lgbm_prev_row["mape_pct"]),
        "lgbm_clima_prev_rmse": float(lgbm_prev_row["rmse_mwmed"]),
        "lgbm_clima_prev_gain_mae_pct": float(lgbm_prev_row["ganho_mae_vs_base_pct"]),
        "lgbm_clima_prev_folds_won": str(lgbm_prev_row["folds_vencidos"]),
        "lgbm_clima_era5_mae": float(lgbm_era5_row["mae_mwmed"]),
        "lgbm_clima_era5_mape": float(lgbm_era5_row["mape_pct"]),
        "lgbm_clima_era5_gain_mae_pct": float(lgbm_era5_row["ganho_mae_vs_base_pct"]),
        "days_lgbm_better_count": int(
            (eval_enriched["err_lgbm"] < eval_enriched["err_base"]).sum()
        ),
        "days_lgbm_better_pct": float(
            (eval_enriched["err_lgbm"] < eval_enriched["err_base"]).mean() * 100.0
        ),
        "extreme_days_count": int(len(extreme_sub)),
        "extreme_days_lgbm_count": int((eval_enriched["err_lgbm"] > 5000.0).sum()),
        "extreme_base_mae": ext_base_mae,
        "extreme_lgbm_mae": ext_lgbm_mae,
        "extreme_gain_pct": (
            ((ext_base_mae - ext_lgbm_mae) / ext_base_mae * 100.0) if ext_base_mae > 0 else 0.0
        ),
        "days_within_2k_base_pct": float((eval_enriched["err_base"] <= 2000.0).mean() * 100.0),
        "days_within_2k_lgbm_pct": float((eval_enriched["err_lgbm"] <= 2000.0).mean() * 100.0),
        "fold5_base_mae": float(fold5_row["baseline_mae_mwmed"]),
        "fold5_lgbm_base_mae": float(fold5_row["lgbm_mae_mwmed"]),
        "fold5_lgbm_clima_prev_mae": float(fold5_row["lgbm_clima_prev_15c_mae"]),
        "fold5_lgbm_clima_era5_mae": float(fold5_row["lgbm_clima_era5_mae"]),
        "fold5_mean_temp": float(fold5_eval["temp_media_se"].mean()),
        "fold5_max_temp": float(fold5_eval["temp_max_se"].max()),
        "fold5_bias_base": float(fold5_eval["res_base"].mean()),
        "fold5_bias_lgbm": float(fold5_eval["res_lgbm"].mean()),
        "other_folds_base_mae": float(other_eval["err_base"].mean()),
        "other_folds_lgbm_mae": float(other_eval["err_lgbm"].mean()),
        "other_folds_gain_pct": float(
            (other_eval["err_base"].mean() - other_eval["err_lgbm"].mean())
            / other_eval["err_base"].mean()
            * 100.0
        ),
    }

    return {
        "overall_summary": overall_summary,
        "day_type_breakdown": day_type_breakdown,
        "error_quantiles": error_quantiles,
        "eval_enriched": eval_enriched,
        "feature_importance_base": fi_base,
        "group_importance_base": group_base,
        "feature_importance_clima": fi_clima,
        "group_importance_clima": group_clima,
        "fold_comparison": fold_comparison,
    }

