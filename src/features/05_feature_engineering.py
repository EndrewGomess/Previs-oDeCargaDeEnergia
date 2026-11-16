#!/usr/bin/env python3
"""05_feature_engineering.py

Engenharia de atributos para previsão de carga diária de energia (horizonte D+7)
sem vazamento temporal (data leakage).

Regras garantidas:
1. Todos os lags e estatísticas móveis utilizam exclusivamente informações de t - 7 dias
   ou mais antigas (lag >= 7).
2. Variáveis de calendário e feriados nacionais brasileiros (holidays) dependem apenas
   da data-alvo (conhecidas antecipadamente).
3. Regime metodológico do ONS codificado explicitamente para refletir a entrada da MMGD.
"""

from __future__ import annotations

from typing import List

import holidays
import numpy as np
import pandas as pd

FEATURE_COLUMNS: List[str] = [
    # Lags estritamente >= 7 dias
    "lag_7",
    "lag_8",
    "lag_9",
    "lag_14",
    "lag_21",
    "lag_28",
    "lag_364",
    # Estatísticas móveis calculadas sobre série defasada em 7 dias
    "rolling_mean_7d_lag7",
    "rolling_std_7d_lag7",
    "rolling_mean_28d_lag7",
    "rolling_std_28d_lag7",
    "diff_lag7_lag14",
    "ratio_lag7_rolling28",
    # Variáveis de calendário
    "dayofweek",
    "is_weekend",
    "day",
    "month",
    "dayofyear",
    "weekofyear",
    "sin_dayofweek",
    "cos_dayofweek",
    "sin_dayofyear",
    "cos_dayofyear",
    # Feriados nacionais brasileiros e contexto
    "is_holiday",
    "is_holiday_eve",
    "is_post_holiday",
    "is_bridge_day",
    # Mudança de metodologia do ONS
    "regime_medicao_code",
]

REGIME_MAP = {
    "1_despachada": 0,
    "2_nao_despachada": 1,
    "3_com_mmgd": 2,
}


def build_features(df_gold: pd.DataFrame) -> pd.DataFrame:
    """Constrói matriz de atributos sem vazamento a partir da série Gold do subsistema SE."""
    df = df_gold.copy()
    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    df = df.sort_values("data_referencia").reset_index(drop=True)

    # Garantia de grade diária contínua por data para evitar deslocamentos por eventual lacuna
    full_dates = pd.date_range(
        start=df["data_referencia"].min(),
        end=df["data_referencia"].max(),
        freq="D",
    )
    df = (
        pd.DataFrame({"data_referencia": full_dates})
        .merge(df, on="data_referencia", how="left")
        .sort_values("data_referencia")
        .reset_index(drop=True)
    )

    target = df["carga_energia_mwmed"]

    # 1. Lags estritamente >= 7 dias
    for lag in [7, 8, 9, 14, 21, 28, 364]:
        df[f"lag_{lag}"] = target.shift(lag)

    # 2. Estatísticas móveis baseadas apenas no passado acessível em t-7
    shifted_7 = target.shift(7)
    df["rolling_mean_7d_lag7"] = shifted_7.rolling(window=7, min_periods=7).mean()
    df["rolling_std_7d_lag7"] = shifted_7.rolling(window=7, min_periods=7).std()
    df["rolling_mean_28d_lag7"] = shifted_7.rolling(window=28, min_periods=28).mean()
    df["rolling_std_28d_lag7"] = shifted_7.rolling(window=28, min_periods=28).std()

    df["diff_lag7_lag14"] = df["lag_7"] - df["lag_14"]
    df["ratio_lag7_rolling28"] = df["lag_7"] / df["rolling_mean_28d_lag7"]

    # 3. Variáveis de calendário e codificação cíclica
    dt = df["data_referencia"].dt
    df["dayofweek"] = dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["day"] = dt.day
    df["month"] = dt.month
    df["dayofyear"] = dt.dayofyear
    df["weekofyear"] = dt.isocalendar().week.astype(int)

    df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # 4. Feriados nacionais brasileiros (conhecidos a priori para qualquer data futura)
    years = sorted(df["data_referencia"].dt.year.unique().tolist())
    # Inclui ano anterior e próximo para cobrir véspera/pós-feriado de virada de ano
    years_extended = list(range(min(years) - 1, max(years) + 2))
    br_holidays = holidays.country_holidays("BR", years=years_extended)

    holiday_dates = set(br_holidays.keys())
    dates_only = df["data_referencia"].dt.date

    df["is_holiday"] = dates_only.isin(holiday_dates).astype(int)
    df["is_holiday_eve"] = (
        (df["data_referencia"] + pd.Timedelta(days=1)).dt.date.isin(holiday_dates).astype(int)
    )
    df["is_post_holiday"] = (
        (df["data_referencia"] - pd.Timedelta(days=1)).dt.date.isin(holiday_dates).astype(int)
    )

    # Dia ponte: segunda-feira antes de feriado na terça, ou sexta-feira após feriado na quinta
    is_mon_before_hol = (df["dayofweek"] == 0) & (df["is_holiday_eve"] == 1)
    is_fri_after_hol = (df["dayofweek"] == 4) & (df["is_post_holiday"] == 1)
    df["is_bridge_day"] = ((is_mon_before_hol | is_fri_after_hol) & (df["is_holiday"] == 0)).astype(
        int
    )

    # 5. Regime metodológico ONS
    if "regime_medicao_ons" in df.columns:
        df["regime_medicao_code"] = df["regime_medicao_ons"].map(REGIME_MAP).fillna(2).astype(int)
    else:
        df["regime_medicao_code"] = np.select(
            [
                df["data_referencia"] < pd.Timestamp("2021-03-01"),
                df["data_referencia"] < pd.Timestamp("2023-04-29"),
            ],
            [0, 1],
            default=2,
        )

    return df
