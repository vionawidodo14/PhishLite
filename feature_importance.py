import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from bert_mlp_train import load_data
import os

if __name__ == "__main__":
    # ------------------------
    # CONFIG
    # ------------------------
    model_path = "mlp_model_best_fold2_URL_Content.pkl"
    phish_dir = "dataset/phishing"
    benign_dir = "dataset/benign"


    # ------------------------
    # Load model
    # ------------------------
    model = joblib.load(model_path)

    # ------------------------
    # Load dataset again (deterministic subset)
    # ------------------------
    random_state = 42
    random.seed(random_state)
    if os.path.exists("X.npy") and os.path.exists("y.npy"):
        print("Loading pre-saved dataset from X.npy and y.npy...")
        X = np.load("X.npy", allow_pickle=True)
        y = np.load("y.npy", allow_pickle=True)
    else:
        print("Generating dataset and saving for future runs...")
        X, y = load_data(phish_dir, benign_dir, sample_size=5000)
        np.save("X.npy", X)
        np.save("y.npy", y)
        print("Saved X.npy and y.npy for future runs.")

    print("Data shape:", X.shape, y.shape)
    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    print("Model accuracy on test set:", accuracy_score(y_test, model.predict(X_test)))

    # ------------------------
    # Feature Names (handcrafted only, last 58)
    # ------------------------
    url_features = [
        "URL_length","URL_IP","URL_redirect","URL_shortener","URL_subdomains","URL_at","URL_fakeHTTPS",
        "URL_dash","URL_dataURI","URL_numberofCommonTerms","URL_checkNumerical","URL_checkPathExtend",
        "URL_checkPunycode","URL_checkSensitiveWord","URL_checkTLDinPath","URL_checkTLDinSub",
        "URL_totalWordUrl","URL_shortestWordUrl","URL_shortestWordHost","URL_shortestWordPath",
        "URL_longestWordUrl","URL_longestWordHost","URL_longestWordPath","URL_averageWordUrl",
        "URL_averageWordHost","URL_averageWordPath","URL_checkStatisticRe"
    ]
    html_features = [
        "HTML_Objects","HTML_metaScripts","HTML_FrequentDomain","HTML_Commonpage","HTML_CommonPageRatioinFooter",
        "HTML_SFH","HTML_popUp","HTML_RightClick","HTML_DomainwithCopyright","HTML_nullLinksinWeb",
        "HTML_nullLinksinFooter","HTML_BrokenLink","HTML_LoginForm","HTML_HiddenInfo_div","HTML_HiddenInfo_button",
        "HTML_HiddenInfo_input","HTML_TitleUrlBrand","HTML_IFrame","HTML_favicon","HTML_statusBarMod",
        "HTML_css","HTML_anchors"
    ]
    rep_features = [
        "REP_SearchEngine","REP_checkGI","REP_pageRank","REP_DNS","REP_registrationLen","REP_Age",
        "REP_abnormal","REP_ports","REP_SSL"
    ]
    handcrafted_feature_names = url_features + html_features + rep_features

    # ------------------------
    # Permutation Importance with progress bar
    # ------------------------
    print("\nComputing permutation importance for handcrafted features...")

    # tqdm wrapper around permutation importance
    def permutation_with_progress(model, X, y, n_repeats, random_state, feature_indices):
        results = np.zeros(len(feature_indices))
        for local_i, feat_idx in enumerate(tqdm(feature_indices, desc="Permuting features", unit="feat")):
            res = permutation_importance(
                model, X, y, n_repeats=n_repeats,
                random_state=random_state, n_jobs=1, scoring="accuracy"
            )
            results[local_i] = res.importances_mean[feat_idx]
        return results

    res = permutation_importance(
        model, X_test, y_test, n_repeats=10,
        random_state=random_state, n_jobs=-1, scoring="accuracy"
    )
    handcrafted_indices = list(range(X.shape[1] - 58, X.shape[1]))
    handcrafted_importances = res.importances_mean[handcrafted_indices]
    importance_map = dict(zip(handcrafted_feature_names, handcrafted_importances))

    print("\nHandcrafted Features by Importance:")
    for feat, score in sorted(importance_map.items(), key=lambda x: x[1], reverse=True):
        print(f"{feat}: {score:.4f}")