#!/usr/bin/env python3
"""01_bronze_download.py

Realiza o download idempotente dos arquivos Parquet de carga diária de energia do ONS
(Operador Nacional do Sistema Elétrico) diretamente do bucket S3 público:
s3://ons-aws-prod-opendata/dataset/carga_energia_di/

Período obrigatório do desafio: 2018 até o ano atual (2026).
Princípio da Camada Bronze: dados brutos imutáveis, salvos sem nenhuma alteração.
"""

from __future__ import annotations

import logging
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingestion.bronze")

BASE_S3_URL = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di"
DEFAULT_START_YEAR = 2018


def get_target_years(start_year: int = DEFAULT_START_YEAR) -> List[int]:
    """Retorna lista de anos desde start_year até o ano corrente."""
    current_year = datetime.now().year
    return list(range(start_year, current_year + 1))


def download_file(url: str, dest_path: Path, max_retries: int = 3, timeout_sec: int = 30) -> bool:
    """Faz download de arquivo via HTTP com retentativas e validação básica."""
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ONS-Carga-Pipeline/1.0 (Bronze Ingestion)"},
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                content = response.read()

            if not content:
                logger.warning(
                    "Resposta vazia recebida de %s (tentativa %d/%d)", url, attempt, max_retries
                )
                time.sleep(1)
                continue

            # Grava temporariamente e renomeia para atomicidade
            tmp_path = dest_path.with_suffix(".tmp")
            tmp_path.write_bytes(content)
            tmp_path.replace(dest_path)
            return True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning("Arquivo não encontrado no S3 (HTTP 404): %s", url)
                return False
            logger.warning(
                "Erro HTTP %d ao baixar %s (tentativa %d/%d): %s",
                e.code,
                url,
                attempt,
                max_retries,
                e,
            )
            time.sleep(2)
        except Exception as e:
            logger.warning("Erro na tentativa %d/%d ao baixar %s: %s", attempt, max_retries, url, e)
            time.sleep(2)

    return False


def run_bronze_ingestion(
    start_year: int = DEFAULT_START_YEAR,
    output_dir: Path = Path("data/bronze"),
    force_download: bool = False,
) -> List[Path]:
    """Executa o download de todos os arquivos Parquet de carga diária de energia."""
    output_dir.mkdir(parents=True, exist_ok=True)
    years = get_target_years(start_year)
    downloaded_files: List[Path] = []

    logger.info("Iniciando ingestão da camada Bronze para os anos: %s", years)
    logger.info("Diretório de destino: %s", output_dir.resolve())

    for year in years:
        filename = f"CARGA_ENERGIA_{year}.parquet"
        dest_file = output_dir / filename
        url = f"{BASE_S3_URL}/{filename}"

        if dest_file.exists() and not force_download:
            size_kb = dest_file.stat().st_size / 1024
            logger.info("Arquivo já existe: %s (%.1f KB) - pulando", filename, size_kb)
            downloaded_files.append(dest_file)
            continue

        logger.info("Baixando %s de %s ...", filename, url)
        success = download_file(url, dest_file)

        if success:
            size_kb = dest_file.stat().st_size / 1024
            logger.info("Download concluído: %s (%.1f KB)", filename, size_kb)
            downloaded_files.append(dest_file)
        else:
            logger.error("Falha ao baixar %s", filename)

    logger.info(
        "Ingestão concluída. Total de arquivos prontos: %d/%d",
        len(downloaded_files),
        len(years),
    )
    return downloaded_files


if __name__ == "__main__":
    force = "--force" in sys.argv
    files = run_bronze_ingestion(force_download=force)
    if not files:
        sys.exit(1)
