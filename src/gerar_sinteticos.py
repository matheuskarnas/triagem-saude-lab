"""
Gerador de dados sintéticos para triagem médica.
Usa Ollama/Mistral com prompt clínico rigoroso para gerar exemplos
balanceados e clinicamente corretos nas 3 classes de urgência.

Uso:
    python3 src/gerar_sinteticos.py
    python3 src/gerar_sinteticos.py --por-classe 150 --output data/raw/dataset_sintetico.csv
"""

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"  # troque por "mistral:7b" se necessário

LABEL_MAP = {
    "LEVE": 0,
    "MODERADO": 1,
    "URGENTE": 2,
}

# Quantos exemplos gerar por classe (padrão)
DEFAULT_POR_CLASSE = 120

# ---------------------------------------------------------------------------
# Critérios clínicos — fonte das definições e exemplos fixos
# ---------------------------------------------------------------------------

CRITERIOS = {
    "LEVE": {
        "definicao": (
            "Sintomas leves que NÃO representam risco imediato. "
            "O paciente pode aguardar consulta agendada (dias ou semanas). "
            "Sem alteração de consciência, sem dor intensa, sem dificuldade respiratória."
        ),
        "exemplos_fixos": [
            "tosse leve há dois dias, sem febre",
            "coriza e espirros, provável resfriado comum",
            "dor de cabeça leve, sem febre, sem rigidez de nuca",
            "cansaço após esforço físico intenso no dia anterior",
            "coceira leve na pele, sem inchaço, sem dificuldade respiratória",
            "dor muscular leve após exercício",
            "azia leve após refeição gordurosa",
            "leve irritação nos olhos, sem perda de visão",
            "pequeno corte superficial, sangramento já controlado",
            "dor de dente leve, sem inchaço na face",
        ],
        "nao_incluir": (
            "NÃO inclua: febre acima de 38.5°C, dor no peito, falta de ar, "
            "perda de consciência, sangramento intenso, dor abdominal intensa, "
            "confusão mental, paralisia, convulsão, vômitos repetidos."
        ),
    },
    "MODERADO": {
        "definicao": (
            "Sintomas moderados que precisam de atenção médica em até 24 horas. "
            "Causam desconforto significativo mas SEM risco imediato de vida. "
            "Pode haver febre moderada, dor intensa localizada, ou sintomas que pioram gradualmente."
        ),
        "exemplos_fixos": [
            "febre de 39°C há 12 horas, dor de garganta intensa, dificuldade para engolir",
            "dor abdominal moderada no lado direito, sem rigidez abdominal",
            "tontura ao levantar, sem desmaio, pressão levemente baixa",
            "vômitos repetidos há 6 horas, sem sangue, ainda consciente e hidratado",
            "dor lombar intensa que irradia para a perna, formigamento",
            "corte profundo que não para de sangrar após 10 minutos de pressão",
            "torção no tornozelo com inchaço importante, dificuldade para apoiar o pé",
            "dor de ouvido intensa com febre de 38.8°C",
            "infecção urinária com febre baixa e dor ao urinar",
            "reação alérgica leve a moderada: urticária sem inchaço na garganta",
        ],
        "nao_incluir": (
            "NÃO inclua: dor no peito com irradiação, falta de ar severa, "
            "perda de consciência, convulsão, paralisia facial, AVC suspeito, "
            "sangramento que não cede, choque anafilático, trauma craniano grave."
        ),
    },
    "URGENTE": {
        "definicao": (
            "Sintomas com RISCO IMEDIATO DE VIDA. Requer atendimento em minutos. "
            "Inclui: parada cardíaca, AVC, obstrução das vias aéreas, choque, "
            "trauma grave, sangramento incontrolável, convulsão ativa, "
            "alteração grave de consciência."
        ),
        "exemplos_fixos": [
            "dor no peito intensa com irradiação para o braço esquerdo e sudorese fria",
            "falta de ar severa em repouso, lábios azulados, incapaz de falar frases completas",
            "queda súbita com perda de consciência, não responde a estímulos",
            "fraqueza súbita no lado direito do corpo, fala embaralhada, suspeita de AVC",
            "convulsão ativa há mais de 2 minutos, não cede",
            "sangramento intenso após ferimento, não controlado com pressão",
            "reação alérgica grave: inchaço na garganta, dificuldade para respirar",
            "queimadura em mais de 20% do corpo ou em face/vias aéreas",
            "trauma craniano com perda de consciência e vômitos em jato",
            "dor abdominal em punhalada com abdome rígido, queda de pressão",
            "bebê sem respirar, coloração arroxeada",
            "tentativa de suicídio com ingestão de medicamentos",
        ],
        "nao_incluir": (
            "NÃO inclua: sintomas leves como resfriado, dor muscular leve, "
            "azia, coceira simples, ou qualquer coisa que possa esperar horas."
        ),
    },
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(classe: str, n: int) -> str:
    c = CRITERIOS[classe]
    exemplos_str = "\n".join(f"- {e}" for e in c["exemplos_fixos"])

    return f"""Você é um médico de pronto-socorro brasileiro especialista em triagem clínica.

Sua tarefa: gerar {n} exemplos DIFERENTES de relatos de sintomas para a classe "{classe}".

DEFINIÇÃO DA CLASSE "{classe}":
{c["definicao"]}

EXEMPLOS REAIS DESTA CLASSE (use como referência de estilo e gravidade):
{exemplos_str}

RESTRIÇÃO CRÍTICA:
{c["nao_incluir"]}

INSTRUÇÕES DE FORMATO:
- Retorne SOMENTE um array JSON válido, sem texto antes ou depois
- Cada elemento é uma string com o relato do paciente
- Escreva em português brasileiro informal, como um paciente descreveria seus sintomas
- Varie: às vezes lista de sintomas separados por vírgula, às vezes frase completa
- Cada relato deve ter entre 5 e 40 palavras
- Os {n} exemplos devem ser CLINICAMENTE DISTINTOS entre si
- NÃO repita os exemplos de referência acima

Responda APENAS com o JSON array. Exemplo de formato:
["sintoma a, sintoma b", "paciente relata dor intensa em...", "febre alta com..."]
"""

# ---------------------------------------------------------------------------
# Chamada ao Ollama
# ---------------------------------------------------------------------------

def chamar_ollama(prompt: str, tentativas: int = 3) -> list[str]:
    for tentativa in range(1, tentativas + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.85,
                        "top_p": 0.95,
                        "num_predict": 2048,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            texto = resp.json()["response"].strip()

            # Extrai o JSON mesmo se vier com texto extra
            match = re.search(r"\[.*\]", texto, re.DOTALL)
            if not match:
                raise ValueError(f"JSON não encontrado na resposta:\n{texto[:300]}")

            exemplos = json.loads(match.group())
            if not isinstance(exemplos, list):
                raise ValueError("Resposta não é uma lista")

            # Filtra strings vazias
            exemplos = [str(e).strip() for e in exemplos if str(e).strip()]
            return exemplos

        except Exception as e:
            print(f"  ⚠️  Tentativa {tentativa}/{tentativas} falhou: {e}")
            if tentativa < tentativas:
                time.sleep(3)

    return []

# ---------------------------------------------------------------------------
# Geração em lotes
# ---------------------------------------------------------------------------

def gerar_classe(classe: str, total: int, lote: int = 20) -> list[dict]:
    """Gera `total` exemplos para uma classe, em lotes de `lote`."""
    resultados = []
    label_num = LABEL_MAP[classe]
    gerados = 0

    print(f"\n{'='*60}")
    print(f"Gerando classe: {classe} (alvo: {total} exemplos)")
    print(f"{'='*60}")

    while gerados < total:
        restante = total - gerados
        n_lote = min(lote, restante)

        print(f"  Lote: pedindo {n_lote} exemplos... ", end="", flush=True)
        prompt = build_prompt(classe, n_lote)
        exemplos = chamar_ollama(prompt)

        if not exemplos:
            print("FALHOU, pulando lote.")
            continue

        # Deduplica contra o que já temos
        textos_existentes = {r["texto"].lower() for r in resultados}
        novos = [e for e in exemplos if e.lower() not in textos_existentes]

        for texto in novos:
            resultados.append({
                "texto": texto,
                "label": classe,
                "label_num": label_num,
            })

        gerados = len(resultados)
        print(f"OK ({len(novos)} novos, total: {gerados})")

        # Pequena pausa para não sobrecarregar o Ollama
        time.sleep(1)

    # Garante exatamente `total` exemplos (pode ter gerado a mais)
    random.shuffle(resultados)
    return resultados[:total]

# ---------------------------------------------------------------------------
# Salvar CSV
# ---------------------------------------------------------------------------

def salvar_csv(dados: list[dict], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["texto", "label", "label_num"])
        writer.writeheader()
        writer.writerows(dados)
    print(f"\n✅ Dataset salvo em: {caminho}")
    print(f"   Total de exemplos: {len(dados)}")


def mostrar_distribuicao(dados: list[dict]) -> None:
    from collections import Counter
    dist = Counter(d["label"] for d in dados)
    print("\nDistribuição final:")
    for label, count in sorted(dist.items()):
        pct = count / len(dados) * 100
        print(f"  {label:12s}: {count:4d} ({pct:.1f}%)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Gerador de dados sintéticos para triagem")
    parser.add_argument(
        "--por-classe",
        type=int,
        default=DEFAULT_POR_CLASSE,
        help=f"Exemplos por classe (padrão: {DEFAULT_POR_CLASSE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/dataset_sintetico.csv",
        help="Caminho do CSV de saída",
    )
    parser.add_argument(
        "--lote",
        type=int,
        default=20,
        help="Tamanho do lote por chamada ao Ollama (padrão: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print("🏥 Gerador de Dados Sintéticos — Triagem Médica")
    print(f"   Modelo Ollama : {OLLAMA_MODEL}")
    print(f"   Por classe    : {args.por_classe}")
    print(f"   Lote          : {args.lote}")
    print(f"   Output        : {args.output}")

    # Verifica se Ollama está acessível
    try:
        requests.get("http://localhost:11434", timeout=5)
    except Exception:
        print("\n❌ Ollama não está acessível em localhost:11434")
        print("   Inicie com: ollama serve")
        return

    todos = []
    for classe in LABEL_MAP:
        dados_classe = gerar_classe(classe, args.por_classe, args.lote)
        todos.extend(dados_classe)

    random.shuffle(todos)
    mostrar_distribuicao(todos)
    salvar_csv(todos, Path(args.output))


if __name__ == "__main__":
    main()
