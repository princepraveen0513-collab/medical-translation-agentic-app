import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from core.retrieval.rag_client import RAGClient

if __name__ == "__main__":
    rag = RAGClient()

    query = "i have chest burns"
    ctx = rag.retrieve_context(query)

    print("\n🩺 Medical Context:")
    for c in ctx["medical"]:
        print(" -", c)

    print("\n🎭 Cultural Context:")
    for c in ctx["cultural"]:
        print(" -", c)
