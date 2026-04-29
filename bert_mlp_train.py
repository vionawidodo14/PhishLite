import os
import numpy as np
import joblib
import torch
from transformers import MobileBertModel, MobileBertTokenizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from extract_features_html import extract_features_phishing
from charset_normalizer import from_path
from tqdm import tqdm
import random
import ast
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from parser_for_training_no_patch import generate_text_representation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from datetime import date
import json
from extract_features_html import FEATURE_TIMINGS

# Paths
bert_dir = './model'
phish_dir = 'dataset/phishing'
benign_dir = 'dataset/benign'

# Load BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = MobileBertTokenizer.from_pretrained(bert_dir)
bert = MobileBertModel.from_pretrained(bert_dir).to(device)
bert.eval()


def read_file_with_fallback(path):
    from charset_normalizer import from_path

    results = from_path(path)
    encoding = results.best().encoding if results and results.best() else None

    # Try detected encoding first
    if encoding:
        try:
            with open(path, "r", encoding=encoding, errors="replace") as f:
                return f.read()
        except UnicodeError:
            pass

    # Fallback encodings
    for fallback_enc in ["utf-8", "utf-16le", "utf-16be", "utf-16", "latin-1"]:
        try:
            with open(path, "r", encoding=fallback_enc, errors="replace") as f:
                print(f"Decoded using fallback: {fallback_enc}")
                return f.read()
        except UnicodeError:
            continue

    print("Failed to decode file with all fallbacks.")
    return None

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

    if content.startswith(("http://", "https://", "www.")):
        return content

    for line in content.splitlines():
        if line.lower().startswith("url:"):
            return line.split(":", 1)[1].strip()
        if line.startswith(("http://", "https://", "www.")):
            return line.strip()

    print(f"Could not find URL in {info_path}")
    return None
def get_url_from_textfile(info_path):
    content = read_file_with_fallback(info_path)
    if not content:
        return None
    return content.strip() or None

def extract_features(sample_dir):
    html_path = os.path.join(sample_dir, 'html.txt')
    info_path = os.path.join(sample_dir, 'info.txt')
    # pagerank_path = os.path.join(sample_dir, 'pagerank.txt')
    if not os.path.exists(html_path) or not os.path.exists(info_path):
        print("DNE")
        return None, None, None

    html = read_file_with_fallback(html_path)
    url = get_url_from_info(info_path)
    # pagerank = read_file_with_fallback(pagerank_path).strip()
    if url is None:
        print(f"No URL found in {info_path}")
        return None, None, None
    if html is None or not html.strip():
        print(f"{html_path} is empty or unreadable.")
        return None, None, None
    # if pagerank is None:
    #     print(f"{pagerank_path} is empty or unreadable.")
    #     return None, None, None

    soup = BeautifulSoup(html, 'html.parser')
    handcrafted_vec = extract_features_phishing(soup, url, feat_type='all')

    prettyHTML = generate_text_representation(html)
    # Get BERT CLS embedding
    try:
        tokens = tokenizer(prettyHTML, return_tensors='pt', truncation=True, max_length=256)
        with torch.no_grad():
            outputs = bert(**{k: v.to(device) for k, v in tokens.items()})
            cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    except Exception as e:
        print(f"BERT encoding failed for {sample_dir}: {e}")
        return None, None, None

    # Concatenated vector
    final_vec = np.concatenate([cls_vec, handcrafted_vec])
    cls_vec = None
    return cls_vec, handcrafted_vec, final_vec





def load_data(phish_dir, benign_dir, sample_size=20000, csv_output_path="features_URL_Content_output.csv"):
    X, y = [], []

    # Prepare output CSV
    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["CLS_Token", "Handcrafted_Feature_Vector", "Concatenated_Vector"])

        # Phishing samples
        print("Loading phishing samples...")
        phish_samples = [s for s in os.listdir(phish_dir) if os.path.isdir(os.path.join(phish_dir, s))]
        random.shuffle(phish_samples)
        phish_samples = phish_samples[:sample_size]

        for sampleid in tqdm(phish_samples):
            sample_dir = os.path.join(phish_dir, sampleid)
            try:
                cls_vec, html_vec, final_vec = extract_features(sample_dir)
            except Exception as e:
                print("Error Found!!!!",e)
                continue
            if final_vec is not None:
                X.append(final_vec)
                y.append(1)
                writer.writerow([
                    # cls_vec.tolist(),
                    html_vec.tolist(),
                    final_vec.tolist()
                ])
            else:
                print("No final")

        # Benign samples
        print("Loading benign samples...")
        benign_samples = [s for s in os.listdir(benign_dir) if os.path.isdir(os.path.join(benign_dir, s))]
        random.shuffle(benign_samples)
        benign_samples = benign_samples[:sample_size]

        for sampleid in tqdm(benign_samples):
            sample_dir = os.path.join(benign_dir, sampleid)
            try:
                cls_vec, html_vec, final_vec = extract_features(sample_dir)
            except:
                print("Error Found!!!!")
                continue
            if final_vec is not None:
                X.append(final_vec)
                y.append(0)
                writer.writerow([
                    # cls_vec.tolist(),
                    html_vec.tolist(),
                    final_vec.tolist()
                ])
            else:
                print("No final")

    return np.array(X), np.array(y)

import csv, ast
import numpy as np

def load_data_from_csv_for_pipeline(
    csv_path,
    n_phish,
    n_benign,
    feature_column="Handcrafted_Feature_Vector"
):
    import csv, ast
    import numpy as np

    X, y = [], []
    total_expected = n_phish + n_benign

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            s = row.get(feature_column)

            # skip malformed rows
            if s is None:
                continue
            s = str(s).strip()
            if s == "" or s.lower() == "none":
                continue

            try:
                vec = ast.literal_eval(s)
            except Exception:
                continue

            if not isinstance(vec, (list, tuple)) or len(vec) == 0:
                continue

            X.append(vec)

            # same labeling logic as your original pipeline
            y.append(1 if len(y) < n_phish else 0)

            if len(y) >= total_expected:
                break

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print(f"[CSV LOADER] Loaded {len(y)} samples | Feature dim = {X.shape[1]}")
    return X, y

if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data_from_csv_for_pipeline(
        "features_URL_Content_output.csv",
        n_phish=20000,
        n_benign=20000
    )
    print(X)
    print(y)
    print("Starting 5-fold training...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_f1 = -1.0
    best_model = None
    best_fold = -1

    accuracies, precisions, recalls, f1s = [], [], [], []

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
        print(f"\nFold {fold}...")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=fold)
        # mlp.fit(X_train, y_train)
        #
        # y_pred = mlp.predict(X_test)
        max_components = 128
        ncomp = min(max_components, X_train.shape[1], X_train.shape[0] - 1)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # normalize
            ('pca', PCA(n_components=ncomp, random_state=42)),  # reduce dimensionality
            ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128),
                                  max_iter=200,
                                  random_state=fold))
        ])


        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Fold {fold} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

        # Track the best model by F1
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_fold = fold

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    # Save the best model
    if best_model:
        today = date.today()
        model_path = f"mlp_with_pca_best_fold{best_fold}_URL_Content_{today}_40k.pkl"
        joblib.dump(best_model, model_path)
        print(f"\nBest PCA+MLP model saved to {model_path} (F1: {best_f1:.4f})")

    # Final average results
    print("\nAverage over 5 folds:")
    print(f"Avg Accuracy: {np.mean(accuracies):.4f}")
    print(f"Avg Precision: {np.mean(precisions):.4f}")
    print(f"Avg Recall: {np.mean(recalls):.4f}")
    print(f"Avg F1 Score: {np.mean(f1s):.4f}")
    pca = best_model.named_steps['pca']
    print("PCA kept", pca.n_components_, "components.")
    print("Explained variance ratio (first 10):", pca.explained_variance_ratio_[:10])
    print("Total variance retained:", pca.explained_variance_ratio_.sum())

    avg_timings = {feat: float(np.mean(times)) for feat, times in FEATURE_TIMINGS.items()}

    with open("feature_timing_stats.json", "w") as fp:
        json.dump(avg_timings, fp, indent=4)

    print("\n=== FEATURE TIMING SUMMARY ===")
    for feat, avg in sorted(avg_timings.items(), key=lambda x: x[1], reverse=True):
        print(f"{feat}: {avg:.6f} sec")
