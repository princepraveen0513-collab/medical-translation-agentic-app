"""
build_all_embeddings.py
-----------------------
Creates clean embeddings for:
  1️⃣ bilingual_medical_clean (from preprocessed JSONL)
  2️⃣ cultural_semantics (from curated JSON)

✅ Uses OpenAI `text-embedding-3-large` (3072-dim)
✅ Stores actual text content in metadata
✅ Batches + rate-limited upsert into Pinecone

Run this once to fully rebuild your embeddings.
"""

import os
import json
import time
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# ================================================================
# Setup
# ================================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-med-hi-en")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ Missing API keys in .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# ================================================================
# Helpers
# ================================================================

def embed_texts(texts, model="text-embedding-3-large"):
    """Generate embeddings for a list of texts."""
    response = client.embeddings.create(model=model, input=texts)
    embeddings = [d.embedding for d in response.data]
    return embeddings


def chunk_text(text, chunk_size=600, overlap=100):
    """Simple character-based chunking with overlap."""
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def batch_upsert(vectors, namespace, batch_size=64, delay=0.5):
    """Upload vectors to Pinecone in batches."""
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"  → Upserted batch of {len(batch)} into '{namespace}'")
        time.sleep(delay)


# ================================================================
# 1️⃣ Bilingual Medical Embeddings
# ================================================================

def build_bilingual_embeddings():
    file_path = os.path.join(DATA_DIR, "preprocessed", "bilingual_clean.jsonl")
    namespace = "bilingual_medical_clean"

    if not os.path.exists(file_path):
        print(f"❌ Missing file: {file_path}")
        return

    print(f"\n🩺 Building embeddings for bilingual medical docs → {namespace}")
    all_chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc.get("id")
            src = doc.get("source_file")

            for lang in ["english", "hindi"]:
                text = doc.get(lang, "")
                if not text.strip():
                    continue

                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "id": f"{doc_id}_{lang}_{i}",
                        "text": chunk,
                        "metadata": {
                            "doc_id": doc_id,
                            "source_file": src,
                            "language": lang,
                            "chunk_index": i,
                            "text": chunk  # ✅ store text
                        }
                    })

    print(f"📄 Prepared {len(all_chunks):,} text chunks.")
    all_vectors = []

    for i in tqdm(range(0, len(all_chunks), 64), desc="Embedding bilingual docs"):
        batch = all_chunks[i:i+64]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)
        vectors = [
            {
                "id": batch[j]["id"],
                "values": embeddings[j],
                "metadata": batch[j]["metadata"]
            }
            for j in range(len(batch))
        ]
        all_vectors.extend(vectors)

    batch_upsert(all_vectors, namespace)

    stats = index.describe_index_stats()
    ns = stats.get("namespaces", {}).get(namespace, {})
    print(f"✅ Finished: {ns.get('vector_count', 0)} vectors in '{namespace}'")


# ================================================================
# 2️⃣ Cultural Semantics Embeddings
# ================================================================

def build_cultural_embeddings():
    file_path = os.path.join(DATA_DIR, "cultural_semantics", "data.json")
    namespace = "cultural_semantics"

    if not os.path.exists(file_path):
        print(f"❌ Missing file: {file_path}")
        return

    print(f"\n🎭 Building embeddings for cultural semantics → {namespace}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    vectors = []

    for e in entries:
        text_parts = [
            e.get("expression_native", ""),
            f"({e.get('expression_translit', '')})",
            e.get("literal_translation", ""),
            e.get("clinical_meaning", ""),
            e.get("cultural_context", ""),
            e.get("category", ""),
            f"Risk flag: {e.get('risk_flag', False)}",
            f"Translation guidance: {e.get('translation_guidelines', '')}"
        ]
        full_text = " ".join([t for t in text_parts if t])

        metadata = {k: v for k, v in e.items() if k != "entries"}
        metadata["text"] = full_text

        vectors.append({
            "id": e.get("id", f"entry_{len(vectors)}"),
            "values": None,  # placeholder, will fill after embedding
            "metadata": metadata
        })

    print(f"📜 Prepared {len(vectors):,} cultural entries for embedding.")

    # Batch embed
    for i in tqdm(range(0, len(vectors), 64), desc="Embedding cultural semantics"):
        batch = vectors[i:i+64]
        texts = [v["metadata"]["text"] for v in batch]
        embeddings = embed_texts(texts)
        for j in range(len(batch)):
            batch[j]["values"] = embeddings[j]
        index.upsert(vectors=batch, namespace=namespace)
        time.sleep(0.5)

    stats = index.describe_index_stats()
    ns = stats.get("namespaces", {}).get(namespace, {})
    print(f"✅ Finished: {ns.get('vector_count', 0)} vectors in '{namespace}'")


# ================================================================
# Main Runner
# ================================================================
if __name__ == "__main__":
    print("\n🚀 Starting full embedding rebuild...\n")
    build_bilingual_embeddings()
    build_cultural_embeddings()

    print("\n🎉 All embeddings rebuilt successfully!\n")
