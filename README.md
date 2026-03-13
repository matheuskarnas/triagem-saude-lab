# 🔬 Triagem Saúde — Laboratório

Pipeline de experimentação do sistema de triagem médica: geração de dados sintéticos, análise exploratória e fine-tuning do BERTimbau.

> Este repositório contém os experimentos. Para rodar a API em produção, veja [`triagem-saude`](https://github.com/matheuskarnas/triagem-saude).

---

## 📁 Estrutura

```
triagem-saude-lab/
├── data/
│   ├── raw/
│   │   ├── dataset_final.csv        ← dataset original traduzido (descartado)
│   │   └── dataset_sintetico.csv    ← dataset gerado pelo Mistral
│   └── processed/
│       └── dataset_balanceado.csv   ← dataset final de treino (380 exemplos)
├── notebooks/
│   └── 01_eda.ipynb                 ← análise exploratória
├── models/
│   ├── bertimbau_triagem/           ← modelo fine-tunado (~417MB, não versionado)
│   ├── classifier.pkl               ← baseline TF-IDF + Logistic Regression
│   ├── vectorizer.pkl               ← vetorizador TF-IDF
│   └── confusion_matrix_bert.png    ← matriz de confusão do BERTimbau
├── reports/
│   ├── distribuicao_classes.png
│   ├── distribuicao_tamanhos.png
│   └── top_palavras_por_classe.png
├── src/
│   ├── gerar_sinteticos.py          ← gerador de dados com Ollama/Mistral
│   ├── train_bert.py                ← fine-tuning do BERTimbau
│   └── train.py                     ← treino do baseline TF-IDF
└── requirements.txt
```

---

## ⚙️ Setup

```bash
git clone https://github.com/matheuskarnas/triagem-saude-lab
cd triagem-saude-lab

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🗂️ Sobre o Dataset

### Por que dados sintéticos?

O ponto de partida foi o dataset público **CSympData** (Rahman et al., Figshare — [doi:10.6084/m9.figshare.28547042.v4](https://doi.org/10.6084/m9.figshare.28547042.v4)), anotado por especialistas em inglês. Após tradução automática para português, dois problemas inviabilizaram seu uso:

- Textos de sintomas repetidos com diagnósticos diferentes dependendo de idade, sexo e tempo do sintoma
- Perda semântica na tradução gerando exemplos clinicamente incorretos

**Decisão:** descartar o dataset traduzido e gerar dados sintéticos do zero com critérios clínicos rigorosos.

### Composição do dataset final

| Classe | Exemplos | Descrição |
|---|---|---|
| LEVE | 120 | Sintomas leves, consulta agendada |
| MODERADO | 120 | Atenção em até 24h |
| URGENTE | 140 | Risco imediato de vida |
| **Total** | **380** | |

A classe URGENTE tem 20 exemplos a mais para compensar a maior criticidade clínica e melhorar o Recall.

---

## 🤖 Gerar Dados Sintéticos

Requer [Ollama](https://ollama.com) instalado e rodando com o modelo Mistral:

```bash
ollama serve          # em um terminal separado
ollama pull mistral
```

```bash
# Gerar dataset com 120 exemplos por classe (padrão)
python3 src/gerar_sinteticos.py

# Personalizar quantidade e destino
python3 src/gerar_sinteticos.py \
  --por-classe 150 \
  --output data/raw/dataset_sintetico.csv \
  --lote 20
```

O gerador usa três camadas de controle por classe para garantir qualidade clínica:
- **Definição precisa** com critérios objetivos
- **Exemplos de referência** que ancoram o nível de gravidade
- **Lista de exclusão explícita** que impede vazamento de sintomas graves em classes leves

---

## 🏋️ Retreinar o BERTimbau

```bash
# Com o dataset padrão
python3 src/train_bert.py

# Personalizando hiperparâmetros
python3 src/train_bert.py \
  --data data/processed/dataset_balanceado.csv \
  --output models/ \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5
```

O script salva automaticamente:
- Modelo fine-tunado em `models/bertimbau_triagem/`
- Matriz de confusão em `models/confusion_matrix_bert.png`
- Relatório de classificação com Recall URGENTE destacado no terminal

**Requisitos de hardware:**
- GPU recomendada (testado em GTX 1060 6GB)
- ~15 minutos com GPU / ~2h sem GPU
- Sem GPU: reduza `--batch_size` para 8

---

## 📊 Resultados dos Experimentos

| Modelo | Acurácia | Recall URGENTE | F1 URGENTE |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 82% | 95% | ~0.88 |
| BERTimbau v1 (labels invertidos — inválido) | 85% | 91%* | — |
| **BERTimbau v2 — produção** ✅ | **87%** | **92.9%** | **0.90** |

*resultado inválido por mapeamento incorreto de classes

O BERTimbau foi escolhido para produção mesmo com Recall 2pp menor que o baseline porque **generaliza melhor para vocabulário não visto no treino** — essencial para um sistema que receberá linguagem natural livre de pacientes reais.

---

## 📦 Copiar modelo para produção

Após retreinar:

```bash
cp -r models/bertimbau_triagem ../triagem-saude/models/
```

---

## 🤖 Uso de IA

| Ferramenta | Como foi usada |
|---|---|
| **Ollama + Mistral** | Geração do dataset sintético com prompt clínico rigoroso |
| **Claude (Anthropic)** | Suporte no desenvolvimento e debugging |
