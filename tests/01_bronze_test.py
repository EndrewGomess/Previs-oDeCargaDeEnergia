"""Tests unitários para a ingestão e integridade da camada Bronze."""

import importlib
from pathlib import Path

bronze_download = importlib.import_module("src.ingestion.01_bronze_download")
bronze_validate = importlib.import_module("src.ingestion.01_bronze_validate")
get_target_years = bronze_download.get_target_years
inspect_bronze_file = bronze_validate.inspect_bronze_file


def test_target_years_range():
    """Valida se a lista de anos inicia em 2018 e cobre até o presente."""
    years = get_target_years(2018)
    assert years[0] == 2018
    assert len(years) >= 9


def test_bronze_files_existence():
    """Garante que todos os arquivos Parquet de 2018 a 2026 foram baixados."""
    bronze_dir = Path("data/bronze")
    for year in range(2018, 2027):
        file_path = bronze_dir / f"CARGA_ENERGIA_{year}.parquet"
        assert file_path.exists(), f"Arquivo {file_path.name} não encontrado."
        assert file_path.stat().st_size > 0, f"Arquivo {file_path.name} está vazio."


def test_bronze_schema_and_integrity_2018_and_2026():
    """Testa integridade dos arquivos limite (2018 e 2026)."""
    bronze_dir = Path("data/bronze")
    for year in [2018, 2026]:
        file_path = bronze_dir / f"CARGA_ENERGIA_{year}.parquet"
        passed, metrics = inspect_bronze_file(file_path)
        assert passed, f"Falha na validação de {file_path.name}: {metrics.get('error')}"
        assert metrics["total_rows"] > 1000
        assert "SE" in metrics["subsistemas"]
