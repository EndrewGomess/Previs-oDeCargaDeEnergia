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


def _resolve_dbt_command() -> list[str]:
    """Localiza o executável do dbt no ambiente virtual (Linux/macOS/Windows) ou PATH."""
    candidates = [
        Path(".venv/bin/dbt"),
        Path(".venv/Scripts/dbt.exe"),
        Path(".venv/Scripts/dbt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate.resolve())]
    return ["dbt"]


def run_dbt_pipeline(dbt_dir: Path = Path("dbt_project")) -> None:
    """Executa dbt run e dbt test dentro do diretório dbt_project."""
    cmd_prefix = _resolve_dbt_command()

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
