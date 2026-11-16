#!/usr/bin/env python3
"""02_silver_gold_dbt.py

Orquestra a execução dos modelos dbt (camada Silver 02_ e camada Gold 03_)
e executa todos os testes de qualidade de dados no DuckDB.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("transformation.dbt")


def run_dbt_pipeline(dbt_dir: Path = Path("dbt_project")) -> None:
    """Executa dbt run e dbt test dentro do diretório dbt_project."""
    dbt_bin = Path(".venv/bin/dbt")
    cmd_prefix = [str(dbt_bin.resolve())] if dbt_bin.exists() else ["dbt"]

    for action in ["run", "test"]:
        logger.info("Executando dbt %s em %s ...", action, dbt_dir.resolve())
        result = subprocess.run(
            [*cmd_prefix, action, "--profiles-dir", "."],
            cwd=dbt_dir,
            check=False,
        )
        if result.returncode != 0:
            logger.error("Falha ao executar dbt %s (código %d)", action, result.returncode)
            sys.exit(result.returncode)

    logger.info("Pipeline dbt (Silver 02_ e Gold 03_) concluído e testado com sucesso!")


if __name__ == "__main__":
    run_dbt_pipeline()
