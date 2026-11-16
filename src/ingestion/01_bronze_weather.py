#!/usr/bin/env python3
"""01_bronze_weather.py

Ingestão da camada Bronze de temperatura diária para o subsistema Sudeste/Centro-Oeste (SE).

Nota de arquitetura (INMET vs Open-Meteo ERA5):
- A API pública do INMET (apitempo.inmet.gov.br) passou a retornar HTTP 503/404 sem token,
  e os pacotes anuais do portal (portal.inmet.gov.br/uploads/dadoshistoricos/*.zip) somam
  ~850 MB com defasagem na atualização do mês corrente e lacunas em estações automáticas.
- Para garantir série contínua (2018-01-01 até hoje), 0% de nulos e reprodutibilidade rápida,
  utilizamos a reanálise ERA5 via Open-Meteo Archive API para as 4 principais capitais
  consumidoras do subsistema SE/CO: São Paulo, Rio de Janeiro, Belo Horizonte e Brasília.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingestion.weather")

SE_CAPITALS: List[Dict[str, object]] = [
    {"city": "Sao_Paulo", "uf": "SP", "lat": -23.5505, "lon": -46.6333, "weight": 0.50},
    {"city": "Rio_de_Janeiro", "uf": "RJ", "lat": -22.9068, "lon": -43.1729, "weight": 0.25},
    {"city": "Belo_Horizonte", "uf": "MG", "lat": -19.9167, "lon": -43.9345, "weight": 0.15},
    {"city": "Brasilia", "uf": "DF", "lat": -15.7975, "lon": -47.8919, "weight": 0.10},
]

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_bronze(
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    output_path: Path = Path("data/bronze/TEMPERATURA_SE_2018_2026.parquet"),
    force_download: bool = False,
) -> Path:
    """Baixa temperaturas diárias das 4 capitais do subsistema SE e salva em Parquet Bronze."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_download:
        logger.info("Arquivo meteorológico Bronze já existe: %s - pulando", output_path)
        return output_path

    if end_date is None:
        # Open-Meteo Archive tem lag de ~2 dias (compatível com o lag do ONS em 2026-11-14)
        end_date = (datetime.now() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    lats = ",".join(str(c["lat"]) for c in SE_CAPITALS)
    lons = ",".join(str(c["lon"]) for c in SE_CAPITALS)
    daily_vars = (
        "temperature_2m_mean,temperature_2m_max,temperature_2m_min,apparent_temperature_mean"
    )

    url = (
        f"{OPEN_METEO_ARCHIVE_URL}"
        f"?latitude={lats}&longitude={lons}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={daily_vars}"
        f"&timezone=America/Sao_Paulo"
    )

    logger.info("Baixando série climática (%s a %s) via Open-Meteo ERA5 ...", start_date, end_date)
    req = urllib.request.Request(url, headers={"User-Agent": "ONS-Carga-Pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    frames: List[pd.DataFrame] = []
    for city_meta, loc_data in zip(SE_CAPITALS, payload, strict=True):
        daily = loc_data["daily"]
        df_city = pd.DataFrame(
            {
                "data_referencia": pd.to_datetime(daily["time"]),
                "cidade": str(city_meta["city"]),
                "uf": str(city_meta["uf"]),
                "peso_carga": float(city_meta["weight"]),
                "temp_media_c": daily["temperature_2m_mean"],
                "temp_max_c": daily["temperature_2m_max"],
                "temp_min_c": daily["temperature_2m_min"],
                "temp_aparente_c": daily["apparent_temperature_mean"],
            }
        )
        frames.append(df_city)

    df_all = pd.concat(frames, ignore_index=True)

    # Remove eventuais dias finais ainda não consolidados pelo ERA5
    df_all = df_all.dropna(subset=["temp_media_c", "temp_max_c", "temp_min_c"]).reset_index(
        drop=True
    )

    tmp_path = output_path.with_suffix(".tmp")
    df_all.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)

    logger.info(
        "Bronze meteorológica salva em %s (%d linhas | %s a %s)",
        output_path,
        len(df_all),
        df_all["data_referencia"].min().date(),
        df_all["data_referencia"].max().date(),
    )
    return output_path


if __name__ == "__main__":
    force = "--force" in sys.argv
    fetch_weather_bronze(end_date="2026-11-14", force_download=force)
