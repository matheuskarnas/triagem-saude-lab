import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns


# ─── Pré-processamento ────────────────────────────────────────────────────────

def limpar_texto(texto: str) -> str:
    """Limpeza básica do texto de sintomas."""
    texto = str(texto).lower().strip()
    texto = texto.replace(",", " ")
    return " ".join(texto.split())  # remove espaços extras


# ─── Avaliação ────────────────────────────────────────────────────────────────

def avaliar_modelo(model, vectorizer, X_test, y_test, classes, output_dir):
    """Gera métricas e salva matriz de confusão."""

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    print("\n📊 Relatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # Recall da classe EMERGENCIA — métrica mais crítica
    idx_emergencia = list(classes).index("EMERGENCIA")
    recall_emerg = recall_score(y_test, y_pred, average=None)[idx_emergencia]
    print(f"🚨 Recall EMERGENCIA: {recall_emerg:.3f}")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=classes, yticklabels=classes)
    plt.title("Matriz de Confusão")
    plt.ylabel("Real")
    plt.xlabel("Predito")
    plt.tight_layout()

    cm_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"✅ Matriz de confusão salva em: {cm_path}")

    return recall_emerg


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Treina classificador de triagem médica")
    parser.add_argument("--data",   default="data/processed/dataset_balanceado.csv")
    parser.add_argument("--output", default="models/")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # ── Carregar dados ────────────────────────────────────────────────────────
    print(f"📂 Carregando {args.data}...")
    try:
        df = pd.read_csv(args.data, sep=";", encoding="utf-8-sig")
        if "texto" not in df.columns:
            df = pd.read_csv(args.data, sep=",")
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return

    print(f"   {len(df)} exemplos | classes: {df['label'].unique()}")

    # ── Pré-processar ─────────────────────────────────────────────────────────
    df["texto_limpo"] = df["texto"].apply(limpar_texto)
    X = df["texto_limpo"]
    y = df["label"]
    classes = ["EMERGENCIA", "URGENTE", "NAO_URGENTE"]

    # ── Split treino/teste ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n   Treino: {len(X_train)} | Teste: {len(X_test)}")

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    print("\n🔤 Vetorizando com TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigramas e bigramas
        max_features=5000,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    # ── Treinar modelo ────────────────────────────────────────────────────────
    print("🤖 Treinando Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # penaliza mais erros em classes menores
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    # ── Avaliar ───────────────────────────────────────────────────────────────
    recall_emerg = avaliar_modelo(model, vectorizer, X_test, y_test, classes, args.output)

    # ── Salvar modelo ─────────────────────────────────────────────────────────
    model_path      = Path(args.output) / "classifier.pkl"
    vectorizer_path = Path(args.output) / "vectorizer.pkl"

    joblib.dump(model,      model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"\n✅ Modelo salvo em:      {model_path}")
    print(f"✅ Vectorizer salvo em:  {vectorizer_path}")
    print(f"\n🎯 Recall EMERGENCIA final: {recall_emerg:.3f}")

    if recall_emerg < 0.80:
        print("⚠️  Recall abaixo de 0.80 — considere ajustar o threshold na API")
    else:
        print("✅ Recall dentro do esperado para produção!")


if __name__ == "__main__":
    main()