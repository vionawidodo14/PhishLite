import os
import ast
import numpy as np
import joblib
import torch
from transformers import MobileBertModel, MobileBertTokenizer
from bs4 import BeautifulSoup
from charset_normalizer import from_path
from extract_features_html import extract_features_phishing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from parser_for_training_no_patch import generate_text_representation
from tqdm import tqdm  # Make sure this is imported
import time
import psutil

# ==== CONFIG ====
bert_dir = './model'
mlp_model_path = 'mlp_model_best_fold1_URL_Content_compressed.pkl'
sample_dir = 'dataset/sample_dir_5000_samples'

# ==== LOAD MODELS ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = MobileBertTokenizer.from_pretrained(bert_dir)
bert = MobileBertModel.from_pretrained(bert_dir).to(device)
bert.eval()
mlp = joblib.load(mlp_model_path)

# ==== READ FILE WITH FALLBACK ====
def read_file_with_fallback(path):
    results = from_path(path)
    encoding = results.best().encoding if results else None

    try:
        if encoding:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
    except UnicodeDecodeError:
        pass

    for fallback_enc in ["utf-16le", "utf-16be", "utf-16", "latin-1"]:
        try:
            with open(path, "r", encoding=fallback_enc) as f:
                print(f"Decoded {path} using fallback: {fallback_enc}")
                return f.read()
        except UnicodeDecodeError:
            continue

    print(f"Failed to decode {path}")
    return None

# ==== PARSE URL FROM info.txt ====
def get_url_from_info(info_path):
    content = read_file_with_fallback(info_path)
    if not content:
        return None

    content = content.strip()
    if not content:
        return None

    try:
        data = ast.literal_eval(content)
        if isinstance(data, dict) and "url" in data:
            return data["url"]
    except Exception:
        pass

    if content.startswith("http"):
        return content

    for line in content.splitlines():
        if line.lower().startswith("url:"):
            return line.split(":", 1)[1].strip()
        if line.startswith("http"):
            return line.strip()

    print(f"Could not find URL in {info_path})")
    return None


# ==== EXTRACT FEATURES FROM SAMPLE FOLDER ====
def extract_features(folder_path):
    html_path = os.path.join(folder_path, 'html.txt')
    info_path = os.path.join(folder_path, 'info.txt')

    if not os.path.exists(html_path) or not os.path.exists(info_path):
        return None, None

    html = read_file_with_fallback(html_path)
    url = get_url_from_info(info_path)

    if not html or not url:
        return None, None

    soup = BeautifulSoup(html, 'html.parser')
    html_vec = extract_features_phishing(soup, url, feat_type='compressed')
    prettyHTML = generate_text_representation(html)

    tokens = tokenizer(prettyHTML, return_tensors='pt', truncation=True, max_length=256)
    with torch.no_grad():
        outputs = bert(**{k: v.to(device) for k, v in tokens.items()})
        cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    final_vec = np.concatenate([cls_vec, html_vec])
    return final_vec, url

# ==== BATCH PREDICTION ====

def predict_on_samples():
    true_labels = []
    pred_labels = []
    failed_samples = 0
    latencies = []
    memory_usage = []
    process = psutil.Process(os.getpid())

    for label in ["phishing", "benign"]:
        label_val = 1 if label == "phishing" else 0
        folder_path = os.path.join(sample_dir, label)
        if not os.path.isdir(folder_path):
            continue

        print(f"\nPredicting for: {label.upper()} samples")
        folders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

        for sample_path in tqdm(folders, desc=f"Processing {label}"):
            try:
                start = time.time()
                vec, url = extract_features(sample_path)
                if vec is not None:
                    pred = mlp.predict([vec])[0]
                    true_labels.append(label_val)
                    pred_labels.append(pred)
                else:
                    tqdm.write(f"[{label}] {sample_path} → Could not extract features")
                    failed_samples += 1
                end = time.time()
                latencies.append(end - start)

                mem_mb = process.memory_info().rss / (1024 * 1024)
                memory_usage.append(mem_mb)
            except Exception as e:
                tqdm.write(f"Error in {sample_path}: {e}")
                failed_samples += 1

    if true_labels:
        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels)
        rec = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)

        print("\nFinal Evaluation Metrics:")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
        print(f"\nFailed to process: {failed_samples} samples")


        # ==== Latency Stats ====
        latencies = np.array(latencies)
        print("\nLatency Statistics (seconds):")
        print(f"Fastest: {np.min(latencies):.3f}")
        print(f"Slowest: {np.max(latencies):.3f}")
        print(f"Median:  {np.median(latencies):.3f}")
        print(f"Mean:    {np.mean(latencies):.3f}")

        memory_usages = np.array(memory_usage)
        print("\nMemory Usage (MB):")
        print(f"Min:    {np.min(memory_usages):.1f} MB")
        print(f"Max:    {np.max(memory_usages):.1f} MB")
        print(f"Median: {np.median(memory_usages):.1f} MB")
        print(f"Mean:   {np.mean(memory_usages):.1f} MB")
    else:
        print("No predictions were made.")

if __name__ == "__main__":
    predict_on_samples()