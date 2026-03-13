"""
train_bert.py
-------------
Fine-tuning do BERTimbau para classificação de triagem médica.
Usa GPU automaticamente se disponível.

Classes (conforme documento do desafio técnico):
  LEVE     (0) — sintomas leves, consulta agendada
  MODERADO (1) — atenção em até 24h
  URGENTE  (2) — risco imediato de vida

Uso:
    python src/train_bert.py
    python src/train_bert.py --epochs 5 --batch_size 16
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Configuração ─────────────────────────────────────────────────────────────

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"  # BERTimbau
CLASSES    = ["LEVE", "MODERADO", "URGENTE"]
LABEL2ID   = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL   = {i: c for i, c in enumerate(CLASSES)}
MAX_LENGTH = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TriagemDataset(Dataset):
    def __init__(self, textos, labels, tokenizer):
        self.encodings = tokenizer(
            textos,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ─── Avaliação ────────────────────────────────────────────────────────────────

def avaliar(model, loader):
    """Avalia o modelo e retorna predições e labels reais."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def salvar_matriz_confusao(y_true, y_pred, output_dir):
    """Salva matriz de confusão como imagem."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds",
        xticklabels=CLASSES, yticklabels=CLASSES
    )
    plt.title("Matriz de Confusão — BERTimbau Triagem")
    plt.ylabel("Real")
    plt.xlabel("Predito")
    plt.tight_layout()
    path = Path(output_dir) / "confusion_matrix_bert.png"
    plt.savefig(path)
    print(f"✅ Matriz de confusão salva em: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning BERTimbau para triagem médica")
    parser.add_argument("--data",       default="data/processed/dataset_balanceado.csv")
    parser.add_argument("--output",     default="models/")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"🖥️  Dispositivo: {DEVICE}")

    # ── Carregar dados ────────────────────────────────────────────────────────
    print(f"\n📂 Carregando {args.data}...")
    try:
        df = pd.read_csv(args.data, sep=";", encoding="utf-8-sig")
        if "texto" not in df.columns:
            df = pd.read_csv(args.data, sep=",")
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return

    df["label_id"] = df["label"].map(LABEL2ID)

    # Checar se há NaN (labels não mapeados)
    nan_count = df["label_id"].isna().sum()
    if nan_count > 0:
        print(f"⚠️  {nan_count} exemplos com label desconhecido — removendo")
        df = df.dropna(subset=["label_id"])

    df["label_id"] = df["label_id"].astype(int)
    print(f"   {len(df)} exemplos | distribuição: {df['label'].value_counts().to_dict()}")

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"].tolist(),
        df["label_id"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label_id"],
    )
    print(f"   Treino: {len(X_train)} | Teste: {len(X_test)}")

    # ── Tokenizer e Datasets ──────────────────────────────────────────────────
    print(f"\n🔤 Carregando tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TriagemDataset(X_train, y_train, tokenizer)
    test_dataset  = TriagemDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size)

    # ── Modelo ────────────────────────────────────────────────────────────────
    print(f"\n🤖 Carregando modelo: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(DEVICE)

    # ── Otimizador e Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # ── Treino ────────────────────────────────────────────────────────────────
    print(f"\n🏋️  Iniciando fine-tuning ({args.epochs} épocas)...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % 10 == 0:
                print(f"   Época {epoch+1}/{args.epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"   ✅ Época {epoch+1} concluída | Loss médio: {avg_loss:.4f}")

    # ── Avaliação ─────────────────────────────────────────────────────────────
    print("\n📊 Avaliando no conjunto de teste...")
    y_pred, y_true = avaliar(model, test_loader)

    print("\n📊 Relatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    recall_urgente = recall_score(y_true, y_pred, average=None)[2]
    print(f"🚨 Recall URGENTE (classe crítica): {recall_urgente:.3f}")

    salvar_matriz_confusao(y_true, y_pred, args.output)

    # ── Salvar modelo ─────────────────────────────────────────────────────────
    model_dir = Path(args.output) / "bertimbau_triagem"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"\n✅ Modelo salvo em: {model_dir}")
    print(f"🎯 Recall URGENTE final: {recall_urgente:.3f}")

    if recall_urgente < 0.80:
        print("⚠️  Recall abaixo de 0.80 — considere mais épocas ou ajuste de threshold")
    else:
        print("✅ Recall dentro do esperado para produção!")


if __name__ == "__main__":
    main()
