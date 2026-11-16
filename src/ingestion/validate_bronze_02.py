#!/usr/bin/env python3
"""02_validate_bronze.py

Valida os arquivos Parquet da camada Bronze de acordo com as especificações do ONS
e com o dicionário de dados (DicionarioDados_Carga_Energia_Diaria.pdf).

Critérios de validação:
1. Arquivos existem para todos os anos esperados (2018 a 2026).
2. Tamanho dos arquivos > 0 bytes.
3. Schema exato:
   - id_subsistema (string)
   - nom_subsistema (string)
   - din_instante (timestamp / datetime)
   - val_cargaenergiamwmed (float / double)
4. Integridade de dados:
   - Sem valores nulos nas colunas obrigatórias
   - val_cargaenergiamwmed > 0 (conforme dicionário: não permite zerado nem negativo)
   - Contém o subsistema alvo 'SE' (Sudeste/Centro-Oeste)
   - Linhas contínuas e consistência cronológica
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import duckdb
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validation.bronze")

EXPECTED_COLUMNS = {
    "id_subsistema",
    "nom_subsistema",
    "din_instante",
    "val_cargaenergiamwmed",
}
EXPECTED_SUBSYSTEMS = {"SE", "S", "NE", "N"}


def inspect_bronze_file(parquet_path: Path) -> Tuple[bool, Dict[str, any]]:
    """Analisa um arquivo Parquet e retorna status de conformidade e métricas."""
    metrics: Dict[str, any] = {"file": parquet_path.name}

    if not parquet_path.exists():
        metrics["error"] = "Arquivo não encontrado"
        return False, metrics

    size_bytes = parquet_path.stat().st_size
    metrics["size_bytes"] = size_bytes
    if size_bytes == 0:
        metrics["error"] = "Arquivo vazio (0 bytes)"
        return False, metrics

    con = duckdb.connect(database=":memory:")
    try:
        # Schema
        schema_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')").df()
        columns_found = set(schema_df["column_name"].tolist())
        metrics["columns"] = list(columns_found)

        missing_cols = EXPECTED_COLUMNS - columns_found
        if missing_cols:
            metrics["error"] = f"Colunas ausentes: {missing_cols}"
            return False, metrics

        # Estatísticas e checagens de qualidade:
        # CAST para DOUBLE pois anos 2018-2021 vieram como VARCHAR no parquet do ONS.
        query = f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(id_subsistema) as non_null_sub,
                COUNT(nom_subsistema) as non_null_nom,
                COUNT(din_instante) as non_null_date,
                COUNT(val_cargaenergiamwmed) as non_null_val,
                MIN(TRY_CAST(val_cargaenergiamwmed AS DOUBLE)) as min_val,
                MAX(TRY_CAST(val_cargaenergiamwmed AS DOUBLE)) as max_val,
                MIN(din_instante) as min_date,
                MAX(din_instante) as max_date,
                COUNT(DISTINCT id_subsistema) as distinct_subsistemas
            FROM read_parquet('{parquet_path}')
        """
        stats = con.execute(query).df().iloc[0].to_dict()
        metrics.update(stats)

        # Checagem de nulos
        total_rows = stats["total_rows"]
        if total_rows == 0:
            metrics["error"] = "Tabela sem registros (0 linhas)"
            return False, metrics

        for col in ["non_null_sub", "non_null_nom", "non_null_date", "non_null_val"]:
            if stats[col] != total_rows:
                metrics["error"] = f"Valores nulos detectados em {col}"
                return False, metrics

        # Checagem de valores zerados ou negativos
        if stats["min_val"] <= 0:
            metrics["error"] = f"Valor inválido de carga: min_val={stats['min_val']} (esperado > 0)"
            return False, metrics

        # Subsistemas presentes
        sub_list = (
            con.execute(f"SELECT DISTINCT id_subsistema FROM read_parquet('{parquet_path}')")
            .df()["id_subsistema"]
            .tolist()
        )
        metrics["subsistemas"] = sub_list

        if "SE" not in sub_list:
            metrics["error"] = "Subsistema 'SE' (Sudeste/Centro-Oeste) não encontrado no arquivo"
            return False, metrics

        return True, metrics

    except Exception as e:
        metrics["error"] = f"Exceção durante leitura: {e}"
        return False, metrics
    finally:
        con.close()


def run_bronze_validation(bronze_dir: Path = Path("data/bronze")) -> bool:
    """Executa a validação de todos os arquivos da camada Bronze."""
    logger.info("Iniciando validação dos arquivos em %s ...", bronze_dir.resolve())
    parquet_files = sorted(bronze_dir.glob("CARGA_ENERGIA_*.parquet"))

    if not parquet_files:
        logger.error("Nenhum arquivo CARGA_ENERGIA_*.parquet encontrado em %s", bronze_dir)
        return False

    all_passed = True
    summary_rows = []

    for file_path in parquet_files:
        passed, info = inspect_bronze_file(file_path)
        if not passed:
            all_passed = False
            logger.error("❌ Falha na validação de %s: %s", file_path.name, info.get("error"))
        else:
            logger.info(
                "✅ %s: %d linhas | Período: %s a %s | Carga SE min=%.1f, max=%.1f MWmed",
                file_path.name,
                info["total_rows"],
                str(info["min_date"])[:10],
                str(info["max_date"])[:10],
                info["min_val"],
                info["max_val"],
            )
        summary_rows.append(info)

    summary_df = pd.DataFrame(summary_rows)
    print("\n--- RESUMO DE VALIDAÇÃO DA CAMADA BRONZE ---")
    cols_to_show = [
        c
        for c in [
            "file",
            "total_rows",
            "min_date",
            "max_date",
            "min_val",
            "max_val",
            "distinct_subsistemas",
        ]
        if c in summary_df.columns
    ]
    print(summary_df[cols_to_show].to_string(index=False))
    print("--------------------------------------------\n")

    if all_passed:
        logger.info("🎉 Todos os arquivos da camada Bronze foram validados com sucesso!")
    else:
        logger.error("⚠️ Alguns arquivos da camada Bronze apresentaram inconsistências.")

    return all_passed


if __name__ == "__main__":
    success = run_bronze_validation()
    sys.exit(0 if success else 1)
