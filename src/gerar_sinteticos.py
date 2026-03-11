"""
gerar_sinteticos.py
-------------------
Gera exemplos sintéticos de sintomas em português para balancear o dataset.
Usa Ollama localmente (gratuito, sem internet após download do modelo).

Uso:
    python gerar_sinteticos.py --input dataset_final.csv --output dataset_balanceado.csv

Requer:
    pip install ollama pandas
    ollama pull ministral-3b  (ou outro modelo)
"""

from html import parser

import ollama
import pandas as pd
import json
import time
import argparse
import random
from collections import Counter

# ─── Configuração ────────────────────────────────────────────────────────────

TARGET_PER_CLASS = 400   # exemplos alvo por classe
BATCH_SIZE       = 15    # exemplos por chamada ao modelo
MODEL_NAME       = "mistral"  # modelo local via Ollama

LABEL_MAP = {
    "URGENTE": {
        "num": 1,
        "descricao": (
            "Sintomas que precisam de atenção médica em até 24h, mas sem risco imediato de morte. "
            "Exemplos: febre alta persistente, dor intensa localizada, tontura com vômito, "
            "dificuldade respiratória leve, corte profundo, crise de asma moderada."
        ),
    },
    "NAO_URGENTE": {
        "num": 0,
        "descricao": (
            "Sintomas leves que podem aguardar consulta agendada. "
            "Exemplos: resfriado comum, dor de garganta leve, cansaço sem causa grave, "
            "coceira leve, dor de cabeça fraca, nariz entupido, tosse seca leve."
        ),
    },
}

# ─── Prompt ──────────────────────────────────────────────────────────────────

def build_prompt(label: str, descricao: str, exemplos_existentes: list, quantidade: int) -> str:
    amostra = random.sample(exemplos_existentes, min(8, len(exemplos_existentes)))
    exemplos_str = "\n".join(f"- {e}" for e in amostra)

    return f"""Você é um especialista em triagem médica. Gere descrições de sintomas em português brasileiro.

CLASSE: {label}
DEFINIÇÃO: {descricao}

EXEMPLOS REAIS DO DATASET:
{exemplos_str}

TAREFA: Gere exatamente {quantidade} novas descrições de sintomas para a classe {label}.

REGRAS:
- Escreva em português brasileiro, letras minúsculas
- Liste sintomas separados por vírgula (sem frases longas)
- Entre 2 e 12 sintomas por descrição
- Varie bastante — não repita combinações dos exemplos acima
- Mantenha coerência clínica com a classe {label}
- NÃO inclua diagnósticos, apenas sintomas

RESPONDA APENAS com JSON válido, sem texto adicional:
{{"exemplos": ["sintoma1, sintoma2", "sintoma3, sintoma4, sintoma5"]}}"""

# ─── Geração ─────────────────────────────────────────────────────────────────

def gerar_exemplos(label: str, info: dict, exemplos_existentes: list, quantidade_total: int) -> list:
    gerados = []
    textos_vistos = set(t.lower().strip() for t in exemplos_existentes)
    tentativas = 0
    max_tentativas = (quantidade_total // BATCH_SIZE + 1) * 4

    print(f"\n  Gerando {quantidade_total} exemplos para [{label}]...")

    while len(gerados) < quantidade_total and tentativas < max_tentativas:
        faltam = quantidade_total - len(gerados)
        batch = min(BATCH_SIZE, faltam + 5)

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": build_prompt(label, info["descricao"], exemplos_existentes, batch)
                }],
                options={"temperature": 0.85, "num_predict": 600}
            )

            raw = response["message"]["content"].strip()

            # Limpar possíveis markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()

            # Extrair só o JSON caso venha com texto antes/depois
            inicio = raw.find("{")
            fim = raw.rfind("}") + 1
            if inicio == -1 or fim == 0:
                raise ValueError("JSON não encontrado na resposta")
            raw = raw[inicio:fim]

            data = json.loads(raw)
            novos = 0

            for texto in data.get("exemplos", []):
                texto = texto.strip().lower()
                if (texto
                        and texto not in textos_vistos
                        and len(texto.split()) >= 2
                        and len(gerados) < quantidade_total):

                    gerados.append({
                        "texto":     texto,
                        "label":     label,
                        "label_num": info["num"]
                    })
                    textos_vistos.add(texto)
                    novos += 1

            print(f"    Lote {tentativas+1}: +{novos} novos | total: {len(gerados)}/{quantidade_total}")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"    ⚠ Erro de parsing no lote {tentativas+1}: {e}")
        except Exception as e:
            print(f"    ⚠ Erro inesperado no lote {tentativas+1}: {e}")
            time.sleep(3)

        tentativas += 1

    if len(gerados) < quantidade_total:
        print(f"    ⚠ Atenção: gerou apenas {len(gerados)}/{quantidade_total} exemplos únicos")
    else:
        print(f"    ✅ [{label}] concluído!")

    return gerados

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Balanceia dataset de triagem com dados sintéticos via Ollama")
    parser.add_argument("--input",  default="dataset_final.csv",      help="CSV de entrada")
    parser.add_argument("--output", default="dataset_balanceado.csv", help="CSV de saída")
    parser.add_argument("--target", type=int, default=TARGET_PER_CLASS, help="Exemplos alvo por classe")
    parser.add_argument("--model",  default="mistral", help="Modelo Ollama a usar")

    args = parser.parse_args()

    global MODEL_NAME
    MODEL_NAME = args.model

    # ── Carregar dataset ──────────────────────────────────────────────────────
    print(f"📂 Carregando {args.input}...")
    try:
        df = pd.read_csv(args.input, sep=';')
        if 'texto' not in df.columns:
            df = pd.read_csv(args.input, sep=',')
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return

    print(f"   {len(df)} linhas carregadas")
    dist = Counter(df['label'])

    print(f"\n📊 Distribuição atual:")
    for k, v in dist.most_common():
        print(f"   {k:15s}: {v:4d} ({v/len(df)*100:.1f}%)")

    # ── Verificar conexão com Ollama ──────────────────────────────────────────
    print(f"\n🤖 Verificando Ollama com modelo '{MODEL_NAME}'...")
    try:
        ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "responda apenas: ok"}]
        )
        print(f"   ✅ Ollama funcionando!")
    except Exception as e:
        print(f"   ❌ Ollama não responde: {e}")
        print(f"   Verifique se o Ollama está rodando: ollama serve")
        print(f"   E se o modelo está baixado: ollama pull {MODEL_NAME}")
        return

    # ── Gerar sintéticos ──────────────────────────────────────────────────────
    todos_sinteticos = []

    for label, info in LABEL_MAP.items():
        atual = dist.get(label, 0)
        faltam = args.target - atual

        if faltam <= 0:
            print(f"\n✅ [{label}] já tem {atual} exemplos — pulando geração")
            continue

        exemplos_existentes = df[df['label'] == label]['texto'].tolist()
        sinteticos = gerar_exemplos(label, info, exemplos_existentes, faltam)
        todos_sinteticos.extend(sinteticos)

    # ── Montar dataset final ──────────────────────────────────────────────────
    df_emergencia = df[df['label'] == 'EMERGENCIA']
    if len(df_emergencia) > args.target:
        print(f"\n✂️  Undersampling EMERGENCIA: {len(df_emergencia)} → {args.target}")
        df_emergencia = df_emergencia.sample(n=args.target, random_state=42)

    df_outros     = df[df['label'] != 'EMERGENCIA']
    df_sinteticos = pd.DataFrame(todos_sinteticos)

    df_final = pd.concat([df_emergencia, df_outros, df_sinteticos], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Salvar ────────────────────────────────────────────────────────────────
    df_final.to_csv(args.output, index=False, sep=';', encoding='utf-8-sig')

    print(f"\n📊 Distribuição final:")
    dist_final = Counter(df_final['label'])
    for k, v in dist_final.most_common():
        print(f"   {k:15s}: {v:4d} ({v/len(df_final)*100:.1f}%)")

    print(f"\n✅ Salvo em: {args.output}")
    print(f"   Total de exemplos: {len(df_final)}")

if __name__ == "__main__":
    main()