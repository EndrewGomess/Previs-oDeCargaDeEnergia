"""Testes unitários e de integração para a camada Silver no DuckDB."""

from pathlib import Path

import duckdb

WAREHOUSE_PATH = Path("data/warehouse.duckdb")


def test_silver_table_exists_and_populated():
    """Garante que silver.silver_carga_energia existe e possui todos os registros esperados."""
    assert WAREHOUSE_PATH.exists(), "Warehouse DuckDB não encontrado."
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    try:
        row_count = con.execute("SELECT COUNT(*) FROM silver.silver_carga_energia").fetchone()[0]
        assert row_count >= 12960, f"Esperado >= 12960 linhas, encontrado {row_count}"

        schema = con.execute("DESCRIBE silver.silver_carga_energia").df()
        types = dict(zip(schema["column_name"], schema["column_type"], strict=False))
        assert types["data_referencia"] == "DATE"
        assert types["carga_energia_mwmed"] == "DOUBLE"
    finally:
        con.close()


def test_silver_no_nulls_or_duplicates():
    """Garante unicidade da chave substituta e ausência de nulos na Silver."""
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    try:
        res = con.execute("""
            SELECT
                COUNT(*) AS total,
                COUNT(DISTINCT sk_carga_diaria) AS unique_keys,
                SUM(CASE WHEN data_referencia IS NULL THEN 1 ELSE 0 END) AS null_dates,
                SUM(CASE WHEN carga_energia_mwmed IS NULL THEN 1 ELSE 0 END) AS null_vals
            FROM silver.silver_carga_energia
        """).fetchone()
        total, unique_keys, null_dates, null_vals = res
        assert total == unique_keys, "Chaves duplicadas detectadas na camada Silver."
        assert null_dates == 0, "Datas nulas detectadas na camada Silver."
        assert null_vals == 0, "Valores de carga nulos detectados na camada Silver."
    finally:
        con.close()
