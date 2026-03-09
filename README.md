# 🔬 triagem-saude-lab

Ambiente de experimentação, treinamento e avaliação de modelos para o sistema de triagem médica.

> Este repositório documenta todo o processo científico por trás do modelo em produção em [triagem-saude](https://github.com/seu-usuario/triagem-saude).

---

## 🗂️ Estrutura

```
triagem-saude-lab/
├── notebooks/
│   ├── 01_eda.ipynb               # Análise exploratória do dataset
│   ├── 02_baseline.ipynb          # TF-IDF + Logistic Regression
│   └── 03_modelo_avancado.ipynb   # BERTimbau fine-tuning
├── src/
│   ├── preprocessing.py           # Limpeza e preparação dos dados
│   ├── gerar_sinteticos.py        # Geração de dados sintéticos (Ollama)
│   ├── train.py                   # Script de treinamento
│   └── evaluate.py                # Métricas e análise de erros
├── tests/
│   ├── test_preprocessing.py
│   └── test_evaluate.py
├── data/
│   ├── raw/                       # Dataset original (não versionado)
│   └── processed/                 # Dataset limpo e balanceado
├── models/                        # Modelos serializados (.pkl)
├── reports/
│   └── relatorio_tecnico.md       # Relatório final
├── requirements.txt
└── README.md
```

---

## 🧪 Experimentos Realizados

| # | Modelo | Acurácia | F1 EMERGENCIA | Recall EMERGENCIA |
|---|--------|----------|---------------|-------------------|
| 1 | TF-IDF + Logistic Regression | — | — | — |
| 2 | TF-IDF + Random Forest | — | — | — |
| 3 | BERTimbau fine-tuned | — | — | — |

> Tabela atualizada conforme experimentos forem concluídos.

---

## 🚀 Como Rodar

### 1. Instale as dependências
```bash
pip install -r requirements.txt
```

### 2. Prepare os dados
```bash
# Coloque o dataset original em data/raw/dataset_final.csv
# Gere o dataset balanceado
python src/gerar_sinteticos.py --input data/raw/dataset_final.csv --output data/processed/dataset_balanceado.csv
```

### 3. Explore os notebooks
```bash
jupyter notebook notebooks/
```

### 4. Treine o modelo
```bash
python src/train.py --data data/processed/dataset_balanceado.csv --output models/
```

### 5. Avalie
```bash
python src/evaluate.py --model models/classifier.pkl --data data/processed/dataset_balanceado.csv
```

---

## 📊 Decisões Técnicas

### Por que dados sintéticos?
O dataset original apresentava desbalanceamento severo (87% EMERGENCIA, 3% URGENTE). Dados sintéticos foram gerados com Ollama (ministral-3b) usando prompt com exemplos reais como referência, garantindo coerência clínica.

### Por que BERTimbau?
Modelo BERT pré-treinado em português brasileiro, incluindo texto informal da internet. Lida melhor com gírias e erros de digitação comuns em relatos de pacientes brasileiros.

### Threshold de decisão
O threshold foi ajustado para priorizar recall da classe EMERGENCIA. Detalhes no [relatório técnico](reports/relatorio_tecnico.md).

---

## 🧪 Testes

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```
