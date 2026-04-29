import os
import sys
import glob
import gc
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    MobileBertForSequenceClassification,
    MobileBertTokenizer,
    Trainer,
    TrainingArguments,
)

# ---------------- CONFIG ----------------
MAX_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 32
MODEL_DIR = "./100kmodel"

# ---------- READ HTML ----------
def read_text(folder_path: str) -> str:
    html_path = os.path.join(folder_path, "html.txt")
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

# ---------- PRETOKENIZATION (ONE-TIME) ----------
def pretokenize(paths, labels, tokenizer, max_length):
    input_ids = []
    attention_masks = []
    y = []

    for path, label in tqdm(zip(paths, labels), total=len(paths), desc="Pre-tokenizing"):
        text = read_text(path)
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids.append(enc["input_ids"])
        attention_masks.append(enc["attention_mask"])
        y.append(label)

    dataset = TensorDataset(
        torch.cat(input_ids),
        torch.cat(attention_masks),
        torch.tensor(y)
    )
    return dataset

def tensor_collator(batch):
    input_ids, attention_masks, labels = zip(*batch)
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.tensor(labels),
    }
# ---------- ARGUMENTS ----------
if len(sys.argv) < 2:
    print("Usage: python script.py <test|full>")
    sys.exit(1)

mode = sys.argv[1]

tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

# ---------- LOAD DATASET PATHS ----------
print("📂 Scanning dataset folders...")
phish_dirs = glob.glob("phreshphish_dataset/Phishing/*")
benign_dirs = glob.glob("phreshphish_dataset/Benign/*")

phishing = [(p, 1) for p in phish_dirs]
benign = [(p, 0) for p in benign_dirs]
data = phishing + benign

if mode == "test":
    data = data[:100]
    print("🧪 Test mode: using 100 samples")

paths = [p for p, y in data]
labels = [y for p, y in data]

print(f"Total samples: {len(paths)}")

# ---------- TRAIN / TEST SPLIT ----------
train_paths, test_paths, train_labels, test_labels = train_test_split(
    paths, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"📊 Train: {len(train_paths)} | Test: {len(test_paths)}")

# ---------- PRETOKENIZE ONCE ----------
train_dataset = pretokenize(train_paths, train_labels, tokenizer, MAX_LENGTH)
test_dataset = pretokenize(test_paths, test_labels, tokenizer, MAX_LENGTH)

gc.collect()

# ---------- MODEL ----------
model = MobileBertForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased",
    num_labels=2
)

# ---------- TRAINING ARGS ----------
args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="no",
    report_to=[],
)

# ---------- TRAINER ----------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=tensor_collator,  # 🔥 REQUIRED
)

# ---------- TRAIN ----------
print("🚀 Training MobileBERT (FAST MODE)...")
trainer.train()

# ---------- EVALUATION ----------
print("🔍 Evaluating...")
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels, preds, average="weighted"
)

accuracy = (preds == torch.tensor(test_labels).numpy()).mean()

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ---------- SAVE MODEL ----------
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"\n💾 Model saved to {MODEL_DIR}")