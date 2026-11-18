#!/usr/bin/env python3
"""slides.py

Ponto de entrada padrão para execução da apresentação executiva em slides via
`streamlit run slides.py`. Delega a renderização completa para `src/app/08_dashboard_slides.py`.
"""

import importlib

dashboard_slides = importlib.import_module("src.app.08_dashboard_slides")
dashboard_slides.main()
