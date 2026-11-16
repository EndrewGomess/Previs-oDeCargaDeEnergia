"""Testes unitários e de integração para a camada Gold (subsistema SE) no DuckDB."""

from pathlib import Path

import duckdb

WAREHOUSE_PATH = Path("data/warehouse.duckdb")


def test_gold_se_only_and_continuous_dates():
    """Valida que a camada Gold contém apenas o subsistema SE e série diária sem lacunas."""
    assert WAREHOUSE_PATH.exists(), "Warehouse DuckDB não encontrado."
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    try:
        subsistemas = (
            con.execute("SELECT DISTINCT id_subsistema FROM gold.gold_carga_energia_se")
            .df()["id_subsistema"]
            .tolist()
        )
        assert subsistemas == ["SE"], f"Subsistema inválido na Gold: {subsistemas}"

        res = con.execute("""
            SELECT
                COUNT(*) AS total_days,
                MIN(data_referencia) AS min_date,
                MAX(data_referencia) AS max_date,
                DATE_DIFF('day', MIN(data_referencia), MAX(data_referencia)) + 1 AS expected_days,
                MIN(carga_energia_mwmed) AS min_load
            FROM gold.gold_carga_energia_se
        """).fetchone()
        total_days, min_date, max_date, expected_days, min_load = res

        assert total_days == expected_days, (
            f"Lacuna detectada na série SE: {total_days} dias vs {expected_days} esperados"
        )
        assert str(min_date) == "2018-01-01"
        assert min_load > 0, f"Valor de carga inválido encontrado: {min_load}"
    finally:
        con.close()
