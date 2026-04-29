import os
import ast
import csv
import json
import random
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
from charset_normalizer import from_path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from extract_features_html import extract_features_phishing, FEATURE_TIMINGS

# ============================================================
# PATHS
# ============================================================

PHISH_DIR = "phreshphish_dataset/Phishing"
BENIGN_DIR = "phreshphish_dataset/Benign"

# ============================================================
# FILE READING WITH FALLBACK
# ============================================================

def read_file_with_fallback(path):
    results = from_path(path)
    enc = results.best().encoding if results and results.best() else None

    if enc:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                return f.read()
        except UnicodeError:
            pass

    for enc in ["utf-8", "utf-16", "utf-16le", "utf-16be", "latin-1"]:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                return f.read()
        except UnicodeError:
            continue

    return None


def get_url_from_textfile(path):
    content = read_file_with_fallback(path)
    return content.strip() if content else None

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
# ============================================================
# FEATURE EXTRACTION (HANDCRAFTED ONLY)
# ============================================================

def extract_handcrafted_features(sample_dir):
    html_path = os.path.join(sample_dir, "html.txt")
    url_path = os.path.join(sample_dir, "url.txt")

    if not os.path.exists(html_path) or not os.path.exists(url_path):
        return None

    html = read_file_with_fallback(html_path)
    url = get_url_from_textfile(url_path)

    if html is None or not html.strip() or url is None:
        return None

    try:
        soup = BeautifulSoup(html, "html.parser")
        feats = extract_features_phishing(soup, url, feat_type="all")
        return feats.astype(np.float32)
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None


# ============================================================
# DATASET LOADER
# ============================================================

def load_dataset(phish_dir, benign_dir, sample_size=2000):
    X, y = [], []

    print("Loading phishing samples...")
    phish_samples = [s for s in os.listdir(phish_dir) if os.path.isdir(os.path.join(phish_dir, s))]
    random.shuffle(phish_samples)
    phish_samples = phish_samples[:sample_size]

    for sid in tqdm(phish_samples):
        vec = extract_handcrafted_features(os.path.join(phish_dir, sid))
        if vec is not None:
            X.append(vec)
            y.append(1)

    print("Loading benign samples...")
    benign_samples = [s for s in os.listdir(benign_dir) if os.path.isdir(os.path.join(benign_dir, s))]
    random.shuffle(benign_samples)
    benign_samples = benign_samples[:sample_size]

    for sid in tqdm(benign_samples):
        vec = extract_handcrafted_features(os.path.join(benign_dir, sid))
        if vec is not None:
            X.append(vec)
            y.append(0)

    return np.array(X), np.array(y)


# ============================================================
# DOFA CORE UTILITIES
# ============================================================

def population_diversity(pop):
    return np.mean(np.abs(pop - np.mean(pop, axis=0)))


# ============================================================
# DOFA FEATURE SELECTION (BINARY)
# ============================================================

def fs_fitness(mask, X, y, alpha=0.95):
    if np.sum(mask) == 0:
        return 1.0

    X_sel = X[:, mask == 1]

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_sel, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=4
    )

    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_va)

    error = 1 - f1_score(y_va, preds)
    size_penalty = np.sum(mask) / X.shape[1]

    return alpha * error + (1 - alpha) * size_penalty


def dofa_feature_selection(X, y, n_agents=8, iters=10, n_replace=2):
    dim = X.shape[1]
    pop = np.random.randint(0, 2, size=(n_agents, dim))
    fitness = np.array([fs_fitness(p, X, y) for p in pop])

    Dt = dim / 2

    for t in range(iters):
        for i in range(n_agents):
            for j in range(n_agents):
                if fitness[j] < fitness[i]:
                    pop[i] = np.where(np.random.rand(dim) < 0.5, pop[j], pop[i])
                    fitness[i] = fs_fitness(pop[i], X, y)

        if population_diversity(pop) < Dt:
            worst = np.argsort(fitness)[-n_replace:]
            pop[worst] = np.random.randint(0, 2, size=(n_replace, dim))
            fitness[worst] = [fs_fitness(p, X, y) for p in pop[worst]]

        Dt *= (1 - t / iters)

    return pop[np.argmin(fitness)]


# ============================================================
# DOFA XGBOOST TUNING (CONTINUOUS)
# ============================================================

PARAM_BOUNDS = np.array([
    [0.05, 0.3],   # learning_rate
    [3, 10],       # max_depth
    [1, 10],       # min_child_weight
    [0.5, 1.0],    # subsample
    [0.5, 1.0],    # colsample_bytree
    [0, 0.5],      # gamma
])


def xgb_fitness(params, X, y):
    lr, md, mcw, subs, col, gam = params
    md = int(round(md))

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=lr,
        max_depth=md,
        min_child_weight=mcw,
        subsample=subs,
        colsample_bytree=col,
        gamma=gam,
        eval_metric="logloss",
        n_jobs=4
    )

    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_va)

    return 1 - f1_score(y_va, preds)


def dofa_xgb_tuning(X, y, n_agents=8, iters=10, n_replace=2):
    lb, ub = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    dim = len(lb)

    pop = np.random.uniform(lb, ub, size=(n_agents, dim))
    fitness = np.array([xgb_fitness(p, X, y) for p in pop])

    Dt = np.mean(ub - lb)

    for t in range(iters):
        for i in range(n_agents):
            for j in range(n_agents):
                if fitness[j] < fitness[i]:
                    beta = np.exp(-np.linalg.norm(pop[i] - pop[j]))
                    pop[i] = np.clip(pop[i] + beta * (pop[j] - pop[i]), lb, ub)
                    fitness[i] = xgb_fitness(pop[i], X, y)

        if population_diversity(pop) < Dt:
            worst = np.argsort(fitness)[-n_replace:]
            pop[worst] = np.random.uniform(lb, ub, size=(n_replace, dim))
            fitness[worst] = [xgb_fitness(p, X, y) for p in pop[worst]]

        Dt *= (1 - t / iters)

    return pop[np.argmin(fitness)]


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset(PHISH_DIR, BENIGN_DIR)

    print("X shape:", X.shape)
    print("Positive ratio:", y.mean())

    print("\nDOFA Feature Selection...")
    mask = dofa_feature_selection(X, y)
    X_sel = X[:, mask == 1]
    print(f"Selected {X_sel.shape[1]} / {X.shape[1]} features")

    print("\nDOFA XGBoost Tuning...")
    params = dofa_xgb_tuning(X_sel, y)
    print("Best params:", params)

    lr, md, mcw, subs, col, gam = params
    md = int(round(md))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sel, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=lr,
        max_depth=md,
        min_child_weight=mcw,
        subsample=subs,
        colsample_bytree=col,
        gamma=gam,
        eval_metric="logloss",
        n_jobs=4
    )

    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]

    print("\nFinal Evaluation")
    print("Accuracy:", accuracy_score(y_te, preds))
    print("Precision:", precision_score(y_te, preds))
    print("Recall:", recall_score(y_te, preds))
    print("F1:", f1_score(y_te, preds))
    print("ROC-AUC:", roc_auc_score(y_te, probs))

    # Save feature timing stats
    avg_timings = {k: float(np.mean(v)) for k, v in FEATURE_TIMINGS.items() if v}
    with open("feature_timing_stats_handcrafted.json", "w") as f:
        json.dump(avg_timings, f, indent=4)

    print("\nFeature timing saved.")