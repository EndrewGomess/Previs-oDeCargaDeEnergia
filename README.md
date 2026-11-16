# Previsão de Carga de Energia Diária (ONS) - Subsistema Sudeste/Centro-Oeste

Projeto de previsão de carga diária de energia elétrica para o subsistema Sudeste/Centro-Oeste (`SE`) do Sistema Interligado Nacional (SIN), com horizonte de 7 dias à frente, utilizando dados abertos do Operador Nacional do Sistema Elétrico (ONS).

---

## ⚡ Pergunta do Desafio

> **O seu modelo prevê melhor do que simplesmente repetir o valor do mesmo dia da semana anterior? Se sim, quanto melhor?**

*(Esta seção será atualizada com os resultados consolidados do backtesting após a execução dos modelos).*

---

## 🏛️ Arquitetura da Solução

O projeto segue a arquitetura em medalhão (Bronze, Silver, Gold) com engenharia de atributos sem vazamento temporal (*leakage*), baseline e modelo de Machine Learning monitorado:

```
├── data/
│   └── bronze/               # Parquets brutos do ONS (2018 até o presente)
├── dbt_project/              # Projeto dbt-duckdb para camadas Silver e Gold
│   ├── models/
│   │   ├── silver/           # Tipagem, padronização e datas validadas
│   │   └── gold/             # Série diária filtrada para subsistema SE e testes de integridade
│   └── dbt_project.yml
├── src/
│   ├── ingestion/            # Pipeline de download e validação bronze
│   ├── features/             # Engenharia de atributos (lags >= 7, feriados, calendário)
│   ├── models/               # Baseline ingênuo e treinamento LightGBM + MLflow
│   └── evaluation/           # Backtesting com janela temporal (MAE, MAPE)
├── pyproject.toml            # Gerenciamento de dependências
├── .gitignore                # Arquivos ignorados pelo Git
└── README.md                 # Documentação do projeto
```

---

## 📋 Etapas do Projeto

1. **Etapa 1 - Ambiente**: Repositório, dependências com `pyproject.toml`, `.gitignore` e documentação inicial.
2. **Etapa 2 - Ingestão (Bronze)**: Download direto dos arquivos Parquet de 2018 até hoje do bucket público ONS S3 (`s3://ons-aws-prod-opendata/dataset/carga_energia_di/`), preservando os dados brutos.
3. **Etapa 3 - Transformação (Silver e Gold)**: Transformações analíticas com `dbt` e `DuckDB`:
   - Silver: padronização de tipos (`DATETIME` -> `DATE`, tipos numéricos).
   - Gold: subsistema Sudeste/Centro-Oeste (`id_subsistema = 'SE'`) com testes dbt (`unique`, `not_null`).
4. **Etapa 4 - Baseline**: Previsão ingênua semanal ($y_{t+7} = y_t$, lag de 7 dias).
5. **Etapa 5 - Modelo Preditivo**: LightGBM com features de calendário, feriados nacionais e lags válidos ($\ge 7$), com rastreamento de experimentos no **MLflow**.
6. **Etapa 6 - Avaliação e Backtesting**: Validação temporal com janela deslizante/expansiva, comparando modelo vs. baseline via **MAE** e **MAPE**.

---

## 🔍 Nuances Metodológicas dos Dados do ONS

Conforme documentado no histórico de evolução da medição de carga do ONS:
- **Até fev/2021**: Carga atendida por usinas despachadas/programadas pelo ONS.
- **Mar/2021 a abr/2023**: Carga despachada + previsão de geração de usinas não despachadas.
- **A partir de 29/04/2023**: Incorporação da estimativa de micro e minigeração distribuída (MMGD).

Essas transições metodológicas são levadas em consideração na engenharia de atributos e na validação dos modelos.

---

## 🚀 Como Executar

### Pré-requisitos
- Python >= 3.11
- `uv` ou `pip`

### 1. Instalação do Ambiente
```bash
# Criar e ativar o ambiente virtual
uv venv
source .venv/bin/activate

# Instalar dependências
uv pip install -e ".[dev]"
```

### 2. Ingestão da Camada Bronze
```bash
python -m src.ingestion.download_bronze
```

### 3. Transformação com dbt (Silver e Gold)
```bash
cd dbt_project
dbt run
dbt test
cd ..
```

### 4. Execução do Baseline, Treinamento e Avaliação
```bash
python -m src.models.train
python -m src.evaluation.backtest
```

### 5. Visualização no MLflow
```bash
mlflow ui
```

---

## 📈 Resultados do Backtesting

| Modelo | MAE (MWmed) | MAPE (%) | Ganho sobre Baseline (%) |
| :--- | :--- | :--- | :--- |
| **Baseline Ingênuo (Semanal)** | *TBD* | *TBD* | 0.0% (Ref.) |
| **LightGBM** | *TBD* | *TBD* | *TBD* |

---

## 🔮 Próximos Passos (com mais tempo)
- Inclusão de dados meteorológicos exógenos (temperatura INMET).
- Modelo global unificado para todos os subsistemas (SE, S, NE, N).
- Dashboard interativo em Streamlit para visualização operacional da previsão e erros.
