#!/usr/bin/env python3
"""04_baseline_semanal.py

Etapa 4: Baseline Ingênuo Semanal em pandas.
Definição: A previsão para o dia t (t+7 em relação à origem da previsão) é igual
ao valor observado no mesmo dia da semana anterior (t - 7 dias).

Restrição contra vazamento (data leakage):
- Utiliza exclusivamente o valor observado 7 dias antes da data-alvo,
  alinhado explicitamente pelo calendário (data_referencia - 7 dias).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("models.baseline")

DEFAULT_WAREHOUSE = Path("data/warehouse.duckdb")


def load_gold_se_data(warehouse_path: Path = DEFAULT_WAREHOUSE) -> pd.DataFrame:
    """Carrega a série diária do subsistema SE da camada Gold no DuckDB."""
    if not warehouse_path.exists():
        raise FileNotFoundError(
            f"Warehouse DuckDB não encontrado em {warehouse_path}. "
            "Execute a Etapa 3 (dbt) primeiro."
        )

    con = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        df = con.execute("""
            SELECT
                data_referencia,
                id_subsistema,
                nom_subsistema,
                carga_energia_mwmed,
                regime_medicao_ons
            FROM gold.gold_carga_energia_se
            ORDER BY data_referencia ASC
        """).df()
    finally:
        con.close()

    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    df = df.sort_values("data_referencia").reset_index(drop=True)
    return df


def compute_weekly_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a previsão ingênua semanal (y_pred(t) = y_obs(t - 7 dias)) em pandas.

    O alinhamento é feito por data exata (t - 7 dias) para garantir ausência de vazamento
    e consistência mesmo se houvesse eventuais dias faltantes.
    """
    result = df.copy()
    result["data_referencia"] = pd.to_datetime(result["data_referencia"])
    result = result.sort_values("data_referencia").reset_index(drop=True)

    # Tabela de lookup com data deslocada +7 dias para representar o valor observado 7 dias antes
    lookup = result[["data_referencia", "carga_energia_mwmed"]].copy()
    lookup["data_referencia"] = lookup["data_referencia"] + pd.Timedelta(days=7)
    lookup = lookup.rename(columns={"carga_energia_mwmed": "y_pred_baseline"})

    result = result.merge(lookup, on="data_referencia", how="left")
    result["erro_absoluto"] = (result["carga_energia_mwmed"] - result["y_pred_baseline"]).abs()
    result["erro_percentual"] = (result["erro_absoluto"] / result["carga_energia_mwmed"]) * 100.0
    return result


def evaluate_baseline_metrics(df_pred: pd.DataFrame) -> Dict[str, float]:
    """Calcula MAE, MAPE (%) e RMSE do baseline sobre os registros com previsão válida."""
    valid = df_pred.dropna(subset=["y_pred_baseline"]).copy()
    if valid.empty:
        raise ValueError("Nenhuma previsão válida disponível para avaliação do baseline.")

    y_true = valid["carga_energia_mwmed"].to_numpy()
    y_pred = valid["y_pred_baseline"].to_numpy()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "n_obs": int(len(valid)),
        "mae_mwmed": mae,
        "mape_pct": mape,
        "rmse_mwmed": rmse,
    }


def run_baseline_pipeline() -> pd.DataFrame:
    """Executa o cálculo do baseline e exibe resumo completo e por ano/regime."""
    logger.info("Carregando série Gold do subsistema SE ...")
    df_gold = load_gold_se_data()

    logger.info("Calculando previsão do baseline ingênuo semanal (lag 7 dias) ...")
    df_baseline = compute_weekly_baseline(df_gold)

    overall = evaluate_baseline_metrics(df_baseline)
    recent_mmgd = evaluate_baseline_metrics(
        df_baseline[df_baseline["regime_medicao_ons"] == "3_com_mmgd"]
    )

    print("\n=== RESULTADOS DO BASELINE INGÊNUO SEMANAL (SE) ===")
    print(
        f"Série Completa (2018-2026) | n={overall['n_obs']} | "
        f"MAE={overall['mae_mwmed']:.2f} MWmed | "
        f"MAPE={overall['mape_pct']:.2f}% | "
        f"RMSE={overall['rmse_mwmed']:.2f} MWmed"
    )
    print(
        f"Regime Atual MMGD (>=2023-04-29) | n={recent_mmgd['n_obs']} | "
        f"MAE={recent_mmgd['mae_mwmed']:.2f} MWmed | "
        f"MAPE={recent_mmgd['mape_pct']:.2f}% | "
        f"RMSE={recent_mmgd['rmse_mwmed']:.2f} MWmed"
    )

    valid = df_baseline.dropna(subset=["y_pred_baseline"]).copy()
    valid["ano"] = valid["data_referencia"].dt.year
    yearly = (
        valid.groupby("ano")
        .agg(
            n_dias=("carga_energia_mwmed", "count"),
            mae_mwmed=("erro_absoluto", "mean"),
            mape_pct=("erro_percentual", "mean"),
        )
        .reset_index()
    )
    print("\n--- Desempenho do Baseline por Ano ---")
    print(yearly.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("====================================================\n")

    return df_baseline


if __name__ == "__main__":
    run_baseline_pipeline()
