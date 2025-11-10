from datasets import load_dataset
import os
from tqdm import tqdm

# ===== CONFIG =====
OUTPUT_DIR = "dataset"
SPLIT = "train"         # choose "train", "test", or "validation"
MAX_PER_CLASS = 500     # e.g. 500 phishing + 500 benign = 1000 total

# ===== FUNCTIONS =====
def save_sample(sample_id, label, html, url):
    """Save one sample into dataset/{Phishing|Benign}/file{n}/html.txt and url.txt"""
    label_dir = "Phishing" if label in [1, "1", "phish", "phishing", True] else "Benign"
    folder = os.path.join(OUTPUT_DIR, label_dir, f"file{sample_id}")
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, "html.txt"), "w", encoding="utf-8", errors="ignore") as f_html:
        f_html.write(html or "")
    with open(os.path.join(folder, "url.txt"), "w", encoding="utf-8", errors="ignore") as f_url:
        f_url.write(url or "")

# ===== MAIN =====
def main():
    dataset = load_dataset("phreshphish/phreshphish", split=SPLIT, streaming=True)
    phishing_count = 0
    benign_count = 0

    for ex in tqdm(dataset, desc=f"Processing {SPLIT} split"):
        html = ex.get("html")
        url = ex.get("url")
        label = ex.get("label")

        # Skip missing fields
        if not html or not url or label is None:
            continue

        # Identify label type
        is_phish = label in [1, "1", "phish", "phishing", True]
        is_benign = label in [0, "0", "benign", "legit", False]

        # Save phishing samples
        if is_phish and phishing_count < MAX_PER_CLASS:
            phishing_count += 1
            save_sample(phishing_count, label, html, url)

        # Save benign samples
        elif is_benign and benign_count < MAX_PER_CLASS:
            benign_count += 1
            save_sample(benign_count, label, html, url)

        # Stop when both reach limit
        if phishing_count >= MAX_PER_CLASS and benign_count >= MAX_PER_CLASS:
            break

    print(f"Saved {phishing_count} phishing and {benign_count} benign samples into '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()