#!/usr/bin/env python3
"""launch_dashboard.py

Launcher automático e multiplataforma (Windows, macOS e Linux) para o Dashboard Streamlit.
Executa apenas com a biblioteca padrão do Python:
1. Garante ambiente virtual (.venv) e instalação das dependências (`pyproject.toml`).
2. Verifica se `data/warehouse.duckdb` e os relatórios em `reports/` estão prontos;
   caso contrário, executa automaticamente ingestão Bronze, clima ERA5 e pipeline dbt.
3. Inicia `streamlit run app.py` em uma porta livre e abre o navegador padrão assim que
   o servidor responder HTTP 200 em `/_stcore/health`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

REQUIRED_REPORTS = [
    "reports/06_backtest_predictions.csv",
    "reports/06_backtest_folds.csv",
    "reports/07_backtest_clima_comparison.csv",
    "reports/07_backtest_clima_folds.csv",
]


def get_venv_python(root_dir: Path = ROOT_DIR) -> Path:
    """Retorna o caminho esperado do interpretador Python dentro de `.venv`."""
    if os.name == "nt":
        return root_dir / ".venv" / "Scripts" / "python.exe"
    return root_dir / ".venv" / "bin" / "python"


def _run_checked(cmd: list[str], cwd: Path = ROOT_DIR, env: dict[str, str] | None = None) -> None:
    """Executa comando exibindo saída em tempo real e falha em caso de erro."""
    print(f"[launcher] Executando: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_environment(root_dir: Path = ROOT_DIR) -> Path:
    """Cria `.venv` se necessário e garante instalação das dependências `[dev]`."""
    venv_python = get_venv_python(root_dir)
    uv_bin = shutil.which("uv")

    if not venv_python.exists():
        print("[launcher] Ambiente virtual .venv não encontrado. Criando...")
        if uv_bin:
            _run_checked([uv_bin, "venv", str(root_dir / ".venv")], cwd=root_dir)
        else:
            _run_checked([sys.executable, "-m", "venv", str(root_dir / ".venv")], cwd=root_dir)

    # Verifica se os pacotes centrais do dashboard já estão instalados
    check_imports = subprocess.run(
        [
            str(venv_python),
            "-c",
            "import streamlit, plotly, duckdb, lightgbm, holidays, dbt.cli.main",
        ],
        cwd=root_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    if check_imports.returncode != 0:
        print("[launcher] Instalando dependências do projeto (.[dev])...")
        if uv_bin:
            _run_checked(
                [uv_bin, "pip", "install", "--python", str(venv_python), "-e", ".[dev]"],
                cwd=root_dir,
            )
        else:
            _run_checked(
                [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], cwd=root_dir
            )
            _run_checked([str(venv_python), "-m", "pip", "install", "-e", ".[dev]"], cwd=root_dir)

    return venv_python


def is_gold_warehouse_ready(venv_python: Path, root_dir: Path = ROOT_DIR) -> bool:
    """Verifica se `data/warehouse.duckdb` existe e possui a tabela Gold climática populada."""
    warehouse_path = root_dir / "data" / "warehouse.duckdb"
    if not warehouse_path.exists():
        return False

    probe = subprocess.run(
        [
            str(venv_python),
            "-c",
            (
                "import duckdb; "
                "con = duckdb.connect('data/warehouse.duckdb', read_only=True); "
                "q = 'SELECT COUNT(*) FROM gold.gold_carga_energia_se_clima'; "
                "n = con.execute(q).fetchone()[0]; "
                "con.close(); "
                "assert n > 0"
            ),
        ],
        cwd=root_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


def ensure_data_pipeline(
    venv_python: Path,
    root_dir: Path = ROOT_DIR,
    force_refresh: bool = False,
) -> None:
    """Executa ingestão Bronze, dbt (Silver/Gold) e backtests caso os dados não existam."""
    if force_refresh or not is_gold_warehouse_ready(venv_python, root_dir):
        print("[launcher] Preparando dados locais (ONS Bronze + Clima ERA5 + dbt DuckDB)...")
        _run_checked([str(venv_python), "src/ingestion/01_bronze_download.py"], cwd=root_dir)
        _run_checked([str(venv_python), "src/ingestion/01_bronze_weather.py"], cwd=root_dir)
        _run_checked([str(venv_python), "src/transformation/02_silver_gold_dbt.py"], cwd=root_dir)

    missing_reports = [p for p in REQUIRED_REPORTS if not (root_dir / p).exists()]
    if force_refresh or missing_reports:
        print("[launcher] Gerando artefatos de backtesting em reports/...")
        _run_checked([str(venv_python), "src/evaluation/06_backtest.py"], cwd=root_dir)
        _run_checked([str(venv_python), "src/evaluation/07_backtest_clima.py"], cwd=root_dir)


def find_free_port(preferred_port: int = 8501, max_attempts: int = 20) -> int:
    """Retorna a primeira porta TCP disponível a partir de `preferred_port`."""
    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(
        f"Nenhuma porta disponível encontrada entre {preferred_port} e "
        f"{preferred_port + max_attempts - 1}."
    )


def wait_for_server(url: str, timeout_seconds: float = 45.0, poll_interval: float = 0.5) -> bool:
    """Aguarda até que o endpoint de saúde do Streamlit responda HTTP 200."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(poll_interval)
    return False


def launch_dashboard(
    port: int = 8501,
    open_browser: bool = True,
    force_refresh: bool = False,
    root_dir: Path = ROOT_DIR,
) -> int:
    """Prepara o ambiente, valida os dados, inicia o Streamlit e abre o navegador."""
    if sys.version_info < (3, 11):
        raise RuntimeError(f"Python >= 3.11 é obrigatório. Versão atual: {sys.version.split()[0]}")

    venv_python = ensure_environment(root_dir)
    ensure_data_pipeline(venv_python, root_dir, force_refresh=force_refresh)

    selected_port = find_free_port(port)
    app_url = f"http://localhost:{selected_port}"
    health_url = f"http://127.0.0.1:{selected_port}/_stcore/health"

    cmd = [
        str(venv_python),
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(selected_port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    print(f"[launcher] Iniciando Dashboard em {app_url} (Ctrl+C para encerrar)...")
    proc = subprocess.Popen(cmd, cwd=root_dir)

    try:
        ready = wait_for_server(health_url, timeout_seconds=45.0)
        if not ready:
            print("[launcher] Aviso: tempo limite atingido aguardando resposta do servidor.")
        elif open_browser:
            print(f"[launcher] Abrindo navegador padrão em {app_url}...")
            webbrowser.open(app_url, new=2)

        return proc.wait()
    except KeyboardInterrupt:
        print("\n[launcher] Encerrando servidor Streamlit...")
        proc.terminate()
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analisa argumentos de linha de comando do launcher."""
    parser = argparse.ArgumentParser(
        description="Launcher automático do Dashboard de Previsão de Carga de Energia (ONS)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Porta inicial preferida para o servidor Streamlit (padrão: 8501).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Inicia o servidor sem abrir o navegador automaticamente.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Reexecuta ingestão Bronze, clima, dbt e backtests antes de abrir o painel.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(
        launch_dashboard(
            port=args.port,
            open_browser=not args.no_browser,
            force_refresh=args.force_refresh,
        )
    )
