import os
import time
import ast
from pathlib import Path
from tqdm import tqdm
from extract_features_html import checkPR  # assumes you already have your checkPR() function

def get_url(sample_dir: Path):
    """Extract URL from url.txt, info.txt, or html.txt."""
    for name in ["url.txt", "info.txt", "html.txt"]:
        path = sample_dir / name
        if not path.exists():
            continue
        text = path.read_text(errors="replace").strip()
        if not text:
            continue

        # Try literal dict (e.g., {'url': 'https://...'})
        try:
            data = ast.literal_eval(text)
            if isinstance(data, dict) and "url" in data:
                return data["url"]
        except Exception:
            pass

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("http://") or line.startswith("https://"):
                return line
            if line.lower().startswith("url:"):
                return line.split(":", 1)[1].strip()
    return None


def process_folder(folder_path: Path):
    """Add pagerank.txt for each subfolder in the dataset, with tqdm progress."""
    sample_dirs = [s for s in folder_path.iterdir() if s.is_dir()]
    print(f"\nProcessing {folder_path} ({len(sample_dirs)} samples)...")

    for sample in tqdm(sample_dirs, desc=f"{folder_path.name} progress", ncols=100):
        pagerank_file = sample / "pagerank.txt"
        if pagerank_file.exists():
            continue  # skip existing

        url = get_url(sample)
        if not url:
            pagerank_file.write_text("NO_URL\n", encoding="utf-8")
            print(f"No URL found in {sample}")
            continue

        try:
            rank = checkPR(url)
        except Exception as e:
            rank = f"ERROR: {e}"

        pagerank_file.write_text(f"{rank}\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path("dataset")
    for category in ["benign", "phishing"]:
        folder = root / category
        if folder.exists():
            process_folder(folder)