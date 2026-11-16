#!/usr/bin/env python3
"""07_feature_engineering_clima.py

Bônus: Engenharia de atributos climáticos (Temperatura Exógena) para previsão D+7.

Separa rigorosamente dois blocos de variáveis climáticas:
1. WEATHER_LAG_FEATURES: Temperaturas observadas em t - 7 dias ou antes (zero previsão futura).
2. WEATHER_FORECAST_FEATURES: Temperatura prevista para o dia-alvo t (disponível via previsão
   numérica de tempo em t - 7 dias). Inclui função auxiliar para simular ruído realista de
   previsão meteorológica D+7 (sigma=1.5 °C) apenas no conjunto de teste.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import List

import duckdb
import numpy as np
import pandas as pd

base_feat_mod = importlib.import_module("src.features.05_feature_engineering")
BASE_FEATURES: List[str] = base_feat_mod.FEATURE_COLUMNS
build_base_features = base_feat_mod.build_features

WEATHER_LAG_FEATURES: List[str] = [
    "temp_media_lag7",
    "temp_max_lag7",
    "temp_aparente_lag7",
    "temp_max_sp_lag7",
    "temp_rolling_mean_7d_lag7",
    "temp_rolling_max_7d_lag7",
    "cdd_lag7",
]

WEATHER_FORECAST_FEATURES: List[str] = [
    "temp_media_target",
    "temp_max_target",
    "temp_min_target",
    "temp_aparente_target",
    "temp_max_sp_target",
    "temp_amplitude_target",
    "cdd_target",
    "cdd_max_target",
    "temp_anomalia_7d",
    "temp_delta_lag7",
]


def load_gold_se_with_weather(
    warehouse_path: Path = Path("data/warehouse.duckdb"),
) -> pd.DataFrame:
    """Carrega a tabela Gold enriquecida com temperatura (gold.gold_carga_energia_se_clima)."""
    if not warehouse_path.exists():
        raise FileNotFoundError(
            f"Warehouse DuckDB não encontrado em {warehouse_path}. Execute o dbt primeiro."
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


def build_weather_features(
    df_gold_clima: pd.DataFrame,
    forecast_noise_std: float = 0.0,
    noise_seed: int = 42,
) -> pd.DataFrame:
    """Constrói features base + features climáticas (lags >= 7 e previsão alvo).

    Args:
        df_gold_clima: DataFrame da tabela gold_carga_energia_se_clima.
        forecast_noise_std: Desvio padrão (°C) do erro simulado de previsão meteorológica D+7.
                            Se > 0, adiciona ruído gaussiano apenas nas colunas de previsão alvo.
        noise_seed: Semente determinística para reprodutibilidade do ruído simulado.
    """
    df = build_base_features(df_gold_clima)

    # 1. Lags climáticos >= 7 dias (100% conhecidos sem previsão futura)
    df["temp_media_lag7"] = df["temp_media_se"].shift(7)
    df["temp_max_lag7"] = df["temp_max_se"].shift(7)
    df["temp_aparente_lag7"] = df["temp_aparente_se"].shift(7)
    df["temp_max_sp_lag7"] = df["temp_max_sp"].shift(7)

    shifted_temp = df["temp_media_se"].shift(7)
    shifted_tmax = df["temp_max_se"].shift(7)
    df["temp_rolling_mean_7d_lag7"] = shifted_temp.rolling(window=7, min_periods=7).mean()
    df["temp_rolling_max_7d_lag7"] = shifted_tmax.rolling(window=7, min_periods=7).max()
    df["cdd_lag7"] = np.maximum(0.0, df["temp_media_lag7"] - 22.0)

    # 2. Temperatura alvo (representa previsão meteorológica D+7 disponível na origem t-7)
    t_mean = df["temp_media_se"].to_numpy(dtype="float64").copy()
    t_max = df["temp_max_se"].to_numpy(dtype="float64").copy()
    t_min = df["temp_min_se"].to_numpy(dtype="float64").copy()
    t_app = df["temp_aparente_se"].to_numpy(dtype="float64").copy()
    t_max_sp = df["temp_max_sp"].to_numpy(dtype="float64").copy()

    if forecast_noise_std > 0.0:
        rng = np.random.default_rng(noise_seed)
        # Erro comum de previsão sinótica na massa de ar + pequeno ruído local
        synoptic_err = rng.normal(0.0, forecast_noise_std, size=len(df))
        t_mean = t_mean + synoptic_err
        t_max = t_max + synoptic_err + rng.normal(0.0, 0.3, size=len(df))
        t_min = t_min + synoptic_err + rng.normal(0.0, 0.3, size=len(df))
        t_app = t_app + synoptic_err
        t_max_sp = t_max_sp + synoptic_err + rng.normal(0.0, 0.4, size=len(df))

    df["temp_media_target"] = t_mean
    df["temp_max_target"] = t_max
    df["temp_min_target"] = np.minimum(t_min, t_max - 0.5)
    df["temp_aparente_target"] = t_app
    df["temp_max_sp_target"] = t_max_sp
    df["temp_amplitude_target"] = df["temp_max_target"] - df["temp_min_target"]

    # Cooling Degree Days (graus-dia de resfriamento -> proxy não-linear de ar-condicionado)
    df["cdd_target"] = np.maximum(0.0, df["temp_media_target"] - 22.0)
    df["cdd_max_target"] = np.maximum(0.0, df["temp_max_target"] - 26.0)
    df["temp_anomalia_7d"] = df["temp_media_target"] - df["temp_rolling_mean_7d_lag7"]
    df["temp_delta_lag7"] = df["temp_media_target"] - df["temp_media_lag7"]

    for col in BASE_FEATURES + WEATHER_LAG_FEATURES + WEATHER_FORECAST_FEATURES:
        df[col] = df[col].astype("float64")

    return df
