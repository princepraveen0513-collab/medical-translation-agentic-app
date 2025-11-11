"""
Improved Retrieval Test
-----------------------
Smartly prints metadata for both:
  - cultural_semantics
  - bilingual_medical_clean
"""

import os
from textwrap import shorten
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# ===============================================================
# Config
# ===============================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ===============================================================
# Helpers
# ===============================================================

def get_query_embedding(query, model="text-embedding-3-large"):
    """Generate embedding for the query text."""
    response = client.embeddings.create(model=model, input=[query])
    emb = response.data[0].embedding
    print(f"\n🧩 Created embedding for query → dim: {len(emb)}")
    return emb


def print_results(title, results):
    print(f"\n🔹 {title}")
    if not results:
        print("   (no results)")
        return
    for i, match in enumerate(results, start=1):
        meta = match.get("metadata", {})
        score = round(match.get("score", 0), 3)

        # Smart metadata printing
        if "category" in meta or "risk_flag" in meta:
            # Cultural semantics
            preview_text = meta.get("expression_native") or meta.get("text", "")
            preview = shorten(preview_text, width=120, placeholder=" ...")
            print(f"   {i}. Score: {score}")
            print(f"      Category: {meta.get('category', 'N/A')}")
            print(f"      Risk Flag: {meta.get('risk_flag', 'N/A')}")
            print(f"      Source: {meta.get('source', 'data.json')}")
            print(f"      Preview: {preview}\n")

        elif "language" in meta:
            # Bilingual medical
            preview_text = meta.get("doc_id") or meta.get("source_file", "")
            preview = shorten(preview_text, width=120, placeholder=" ...")
            print(f"   {i}. Score: {score}")
            print(f"      Source: {meta.get('source_file', 'Unknown')} ({meta.get('language', '-')})")
            print(f"      Preview: {preview}\n")

        else:
            # Unknown structure fallback
            print(f"   {i}. Score: {score}")
            print(f"      Metadata: {meta}\n")


def search_namespace(query, namespace, top_k=3):
    """Search Pinecone namespace."""
    query_vector = get_query_embedding(query)
    return index.query(
        vector=query_vector,
        namespace=namespace,
        top_k=top_k,
        include_metadata=True
    )["matches"]


# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":
    query = input("\n💬 Enter a query (Hindi or English): ").strip()
    if not query:
        query = "सीने में जलन"

    print(f"\n🧠 Query: {query}")

    # Query cultural semantics
    results_cultural = search_namespace(query, "cultural_semantics", top_k=3)
    print_results("Cultural Semantics Results (Idioms / Expressions)", results_cultural)

    # Query bilingual medical (clean)
    results_medical = search_namespace(query, "bilingual_medical_clean", top_k=3)
    print_results("Bilingual Medical Clean Results (Educational Context)", results_medical)
