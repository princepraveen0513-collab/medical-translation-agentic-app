"""
Preprocess a bilingual medical PDF (single test version).
Extracts all English and all Hindi text separately, cleans and normalizes,
and outputs one structured JSON record ready for embedding.
"""

import os
import re
import json
import unicodedata
import fitz  # PyMuPDF
from tqdm import tqdm

# ===============================================================
# Configuration
# ===============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF_PATH = os.path.join(BASE_DIR, "data", "bilingual", "24HourUrine_Hindi.pdf")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "24HourUrine_clean_test.jsonl")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ===============================================================
# Helpers
# ===============================================================

def normalize_text(text: str) -> str:
    """Clean, normalize, lowercase, and remove extra noise."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("–", "-").replace("•", "-")
    noise_patterns = [
        r'www\.[A-Za-z0-9./_-]+',
        r'reproductive health access project',
        r'healthinfotranslations\.org',
        r'page \d+ of \d+',
        r'©.*\d{4}',  # copyright
    ]
    for pat in noise_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    return text.strip()


def detect_language(text: str) -> str:
    """Detect if a text block is English or Hindi."""
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"
    elif re.search(r'[A-Za-z]', text):
        return "en"
    else:
        return "other"


def extract_text_by_language(pdf_path: str):
    """Extract text and separate all English and Hindi content."""
    doc = fitz.open(pdf_path)
    english_blocks, hindi_blocks = [], []

    for page in tqdm(doc, desc="Extracting text"):
        blocks = page.get_text("blocks")
        for b in blocks:
            _, _, _, _, text, *_ = b
            if not text or len(text.strip()) < 3:
                continue
            text = normalize_text(text)
            lang = detect_language(text)
            if lang == "en":
                english_blocks.append(text)
            elif lang == "hi":
                hindi_blocks.append(text)
    doc.close()
    return " ".join(english_blocks), " ".join(hindi_blocks)


# ===============================================================
# Main pipeline
# ===============================================================
def process_pdf(pdf_path, output_path):
    print(f"\n📘 Processing file: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found at {pdf_path}")

    english_text, hindi_text = extract_text_by_language(pdf_path)

    print(f"   → English chars: {len(english_text):,}")
    print(f"   → Hindi chars: {len(hindi_text):,}")

    topic = os.path.splitext(os.path.basename(pdf_path))[0].replace("_Hindi", "")
    record = {
        "id": topic.lower(),
        "source_file": os.path.basename(pdf_path),
        "english_char_count": len(english_text),
        "hindi_char_count": len(hindi_text),
        "english": english_text,
        "hindi": hindi_text,
        "metadata": {
            "extracted_from": pdf_path,
            "processing_stage": "preprocessed_single",
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved processed JSON to: {output_path}\n")


# ===============================================================
# Run
# ===============================================================
if __name__ == "__main__":
    process_pdf(PDF_PATH, OUTPUT_PATH)
