#!/usr/bin/env python3
"""app.py

Ponto de entrada padrão para execução do dashboard via `streamlit run app.py`.
Delega a renderização completa para o módulo `src/app/08_dashboard_app.py`.
"""

import importlib

dashboard_app = importlib.import_module("src.app.08_dashboard_app")
dashboard_app.main()
