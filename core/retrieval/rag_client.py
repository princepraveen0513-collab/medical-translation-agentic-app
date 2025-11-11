import os
from typing import Dict, List, Any

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from textwrap import shorten

load_dotenv()

# ------------ ENV ------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-med-hi-en")

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env")
if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in .env")

# ------------ Clients ------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


class RAGClient:
    """
    RAG client for retrieving:
      - Medical context from `bilingual_medical_clean`
      - Cultural nuance from `cultural_semantics`

    Returns clean text snippets suitable for prompts.
    """

    def __init__(
        self,
        medical_namespace: str = "bilingual_medical_clean",
        cultural_namespace: str = "cultural_semantics",
        top_k_medical: int = 4,
        top_k_cultural: int = 3,
    ):
        self.medical_ns = medical_namespace
        self.cultural_ns = cultural_namespace
        self.top_k_medical = top_k_medical
        self.top_k_cultural = top_k_cultural

    # -------------------------------------------------
    def _embed(self, text: str) -> List[float]:
        """Create embedding for query text."""
        resp = openai_client.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=[text],
        )
        return resp.data[0].embedding

    # -------------------------------------------------
    def _query_namespace(
        self,
        vector: List[float],
        namespace: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Query a specific namespace; return matches (may be empty)."""
        try:
            res = index.query(
                namespace=namespace,
                vector=vector,
                top_k=top_k,
                include_metadata=True,
            )
            return res.get("matches", []) or []
        except Exception as e:
            print(f"⚠️ Query failed for namespace '{namespace}': {e}")
            return []

    # -------------------------------------------------
    def _format_medical(self, matches: List[Dict[str, Any]]) -> List[str]:
        """
        Format bilingual_medical_clean matches.

        We now EXPECT metadata["text"] to be present for each chunk.
        We also append minimal provenance for transparency.
        """
        snippets: List[str] = []

        for m in matches:
            meta = m.get("metadata", {}) or {}
            chunk_text = meta.get("text") or m.get("text") or ""

            if not chunk_text:
                # Fallback (should be rare with correct indexing)
                src = meta.get("source_file") or meta.get("doc_id") or "unknown source"
                snippets.append(f"(No text) from {src}")
                continue

            src = meta.get("source_file") or meta.get("doc_id") or ""
            lang = meta.get("language") or ""
            prefix = ""
            if src or lang:
                prefix = f"[{src} | {lang}] "

            formatted = prefix + shorten(chunk_text, width=400, placeholder=" ...")
            snippets.append(formatted)

        return snippets

    # -------------------------------------------------
    def _format_cultural(self, matches: List[Dict[str, Any]]) -> List[str]:
        """
        Format cultural_semantics matches.

        Primary: use metadata["text"] (rich combined view from preprocessing).
        Secondary: reconstruct from fields if needed.
        """
        snippets: List[str] = []

        for m in matches:
            meta = m.get("metadata", {}) or {}

            # Prefer precomputed full text if present
            full_text = meta.get("text")

            if full_text:
                snippets.append(shorten(full_text, width=400, placeholder=" ..."))
                continue

            # Fallback: build from individual fields
            expr = meta.get("expression_native") or meta.get("expression_translit")
            literal = meta.get("literal_translation")
            clinical = meta.get("clinical_meaning")
            category = meta.get("category")
            risk = meta.get("risk_flag")
            guidance = meta.get("translation_guidelines") or meta.get("guidance")

            parts = []
            if expr:
                parts.append(f"Expression: {expr}")
            if literal:
                parts.append(f"Literal: {literal}")
            if clinical:
                parts.append(f"Clinical: {clinical}")
            if category:
                parts.append(f"Category: {category}")
            if isinstance(risk, bool):
                parts.append(f"Risk flag: {risk}")
            if guidance:
                parts.append(f"Guidance: {guidance}")

            if parts:
                snippets.append(shorten(" | ".join(parts), width=400, placeholder=" ..."))

        return snippets

    # -------------------------------------------------
    def retrieve_context(self, query_text: str) -> Dict[str, List[str]]:
        """
        Main entrypoint:
          - Embeds query
          - Queries both namespaces
          - Returns:
              {
                "medical": [ ...strings... ],
                "cultural": [ ...strings... ]
              }
        """
        if not query_text or not query_text.strip():
            return {"medical": [], "cultural": []}

        emb = self._embed(query_text)

        med_matches = self._query_namespace(
            emb, self.medical_ns, self.top_k_medical
        )
        cult_matches = self._query_namespace(
            emb, self.cultural_ns, self.top_k_cultural
        )

        medical_ctx = self._format_medical(med_matches)
        cultural_ctx = self._format_cultural(cult_matches)

        print(
            f"\n🔍 RAGClient: Retrieved {len(medical_ctx)} medical, "
            f"{len(cultural_ctx)} cultural snippets."
        )

        return {
            "medical": medical_ctx,
            "cultural": cultural_ctx,
        }
