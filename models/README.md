# Models

Os arquivos de modelo (.pkl, .safetensors) não são versionados por serem grandes.

## Como regenerar

### Baseline (TF-IDF + Logistic Regression)
```bash
python src/train.py
```

### BERTimbau fine-tuned
```bash
python src/train_bert.py
```
