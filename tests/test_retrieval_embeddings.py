import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from textwrap import shorten

# ===============================
# Load environment
# ===============================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-med-hi-en")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ Missing API keys in .env")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ===============================
# Helper functions
# ===============================
def embed_text(text: str):
    """Create an embedding vector for a query."""
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
    return resp.data[0].embedding

def show_results(title, matches):
    print(f"\n=== {title} ===")
    if not matches:
        print("(no results found)")
        return
    for i, m in enumerate(matches, start=1):
        text = m["metadata"].get("text", m.get("text", "(no text field)"))
        source = m["metadata"].get("source", "unknown")
        score = round(m["score"], 3)
        print(f"\n{i}. Score: {score}")
        print(f"   Source: {source}")
        print(f"   Text Preview: {shorten(text, width=250, placeholder=' ...')}")


# ===============================
# Run a retrieval query
# ===============================
query = input("\n💬 Enter a test symptom or phrase (Hindi or English): ").strip()
query_emb = embed_text(query)

# Search both namespaces
namespaces = ["bilingual_medical_clean", "cultural_semantics"]
for ns in namespaces:
    print(f"\n🔍 Searching namespace: {ns}")
    results = index.query(
        namespace=ns,
        vector=query_emb,
        top_k=3,
        include_metadata=True
    )["matches"]
    show_results(f"Top Matches from {ns}", results)
