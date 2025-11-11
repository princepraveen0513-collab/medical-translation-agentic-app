import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from textwrap import shorten

# =====================================
# Load env
# =====================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# =====================================
# Helper
# =====================================
def embed_query(text: str):
    res = client.embeddings.create(model="text-embedding-3-large", input=text)
    return res.data[0].embedding

def print_results(namespace, matches):
    print(f"\n🔍 Top matches from namespace: {namespace}\n")
    for i, m in enumerate(matches, start=1):
        md = m.get("metadata", {})
        txt = md.get("text") or m.get("text") or ""
        print(f"{i}. Score: {round(m['score'],3)}")
        print(f"   → Source: {md.get('source_file', md.get('doc_id', 'unknown'))}")
        print(f"   → Lang: {md.get('language','n/a')}")
        print(f"   → Preview: {shorten(txt, width=250, placeholder=' ...')}\n")

# =====================================
# Query
# =====================================
query = input("\n💬 Enter a medical or colloquial Hindi phrase: ").strip()
q_emb = embed_query(query)

for ns in ["bilingual_medical_clean", "cultural_semantics"]:
    print(f"\n🔹 Searching {ns} ...")
    res = index.query(
        namespace=ns,
        vector=q_emb,
        top_k=3,
        include_metadata=True
    )["matches"]
    print_results(ns, res)
