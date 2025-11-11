import uuid
from typing import Dict, Any, Optional

from core.pii.pii_agent import PIIAnonymizer
from core.retrieval.rag_client import RAGClient
from core.agents.translation_agent import TranslationAgent
from core.agents.intent_classifier import IntentClassifier
from core.db.session_manager import SessionManager


class CoordinatorAgent:
    """
    Central orchestrator for the Agentic RAG workflow.

    Responsibilities:
      - Manage sessions
      - Run PII anonymization
      - Use IntentClassifier to decide if medical RAG retrieval is needed
      - Retrieve medical + cultural context via RAG (conditionally)
      - Call TranslationAgent with context
      - Persist everything via SessionManager
      - Maintain persistent memory using automatic summaries
    """

    def __init__(self):
        self.pii = PIIAnonymizer()
        self.rag = RAGClient()
        self.translator = TranslationAgent()
        self.sessions = SessionManager()
        self.intent_classifier = IntentClassifier()  # 🧠 new mini LLM classifier

    # -------------------------------------------------
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Create or register a new conversation session."""
        if not session_id:
            session_id = str(uuid.uuid4())
        self.sessions.create_session(session_id)
        print(f"🆕 Started session: {session_id}")
        return session_id

    # -------------------------------------------------
    def process_message(
        self,
        text: str,
        speaker: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main entrypoint for each message."""
        if not text or not text.strip():
            raise ValueError("Empty message passed to CoordinatorAgent.")

        if not session_id:
            session_id = self.start_session()

        print(f"\n🩺 [Session {session_id}] Message from {speaker}")
        print(f"   Original: {text}")

        # 1️⃣ PII anonymization
        pii_result = self.pii.deidentify(text)
        deid = pii_result.deidentified_text
        print(f"🛡️  De-identified: {deid}")

        # 2️⃣ Intent classification – decide if medical RAG is needed
        intent_result = self.intent_classifier.classify_intent(deid)
        label = intent_result["label"]
        conf = intent_result["confidence"]

        # Log classification outcome to database
        self.sessions.save_medical_rag_reflexion(session_id, deid, label, conf)

        # 3️⃣ Conditional context retrieval
        medical_ctx = []
        cultural_ctx = []

        if label in ["medical_required"]:
            print("🩺 Medical context required → Running RAG retrieval")
            contexts = self.rag.retrieve_context(deid)
            medical_ctx = contexts.get("medical", [])
            cultural_ctx = contexts.get("cultural", [])
        else:
            print(f"🚫 Skipping medical RAG retrieval ({label})")
            # Always still fetch cultural context, lightweight
            contexts = self.rag.retrieve_context(deid)
            medical_ctx = ["Medical context: not required (small talk / non-clinical)."]
            cultural_ctx = contexts.get("cultural", [])

        print(f"🔍 Context Retrieved → Medical: {len(medical_ctx)}, Cultural: {len(cultural_ctx)}")

        # 🧠 Added: Retrieve existing conversation summary (persistent memory)
        conversation_summary = self.sessions.get_summary(session_id)
        if conversation_summary:
            print("🧠 Existing summary found → Injecting into translation.")
        else:
            print("⚠️ No prior summary → proceeding without memory context.")

        # 4️⃣ Translation with context (+ memory)
        t_result = self.translator.translate_with_context(
            deid,
            medical_context=medical_ctx,
            cultural_context=cultural_ctx,
            conversation_summary=conversation_summary,  # 🧠 Added
        )

        translation = t_result["translation"]
        prompt_used = t_result["prompt"]
        direction = t_result["direction"]

        print(f"🌐 Translation ({direction}): {translation}")

        # 5️⃣ Persist message (with full context + intent info)
        record = {
            "session_id": session_id,
            "speaker": speaker,
            "original": text,
            "deidentified": deid,
            "translation": translation,
            "context": {
                "intent_decision": label,
                "intent_confidence": conf,
                "medical_context": medical_ctx,
                "cultural_context": cultural_ctx,
                "llm_prompt": prompt_used,
                "direction": direction,
            },
        }

        self.sessions.save_message(
            session_id=session_id,
            speaker=speaker,
            original=text,
            deidentified=deid,
            translation=translation,
            context=record["context"],
        )

        # 🧠 Added: Auto-summarize once session has ≥ 2 messages
        message_count = self.sessions.count_messages(session_id)
        if message_count >= 2:
            print(f"🧾 Session has {message_count} messages → Generating/Updating summary.")
            try:
                summary = self.summarize_session(session_id)
                print(f"✅ Summary updated for session {session_id}")
            except Exception as e:
                print(f"⚠️ Summary generation failed: {e}")

        # 6️⃣ Structured response for UI
        return {
            "session_id": session_id,
            "original_text": text,
            "deidentified_text": deid,
            "entities": [e.__dict__ for e in pii_result.entities],
            "intent_label": label,
            "intent_confidence": conf,
            "contexts": {
                "medical": medical_ctx,
                "cultural": cultural_ctx,
            },
            "translation": translation,
            "direction": direction,
            "llm_prompt": prompt_used,
        }

    # -------------------------------------------------
    def summarize_session(self, session_id: str, llm_client=None, model: str = None) -> str:
        """Optional: Summarize a full session using an LLM."""
        from openai import OpenAI
        from dotenv import load_dotenv
        import os

        if llm_client is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY for summarization.")
            llm_client = OpenAI(api_key=api_key)

        if model is None:
            model = os.getenv("OPENAI_MODEL", "gpt-4o")

        conv = self.sessions.get_conversation(session_id)
        if not conv:
            raise ValueError(f"No messages found for session {session_id}")

        convo_text = ""
        for m in conv:
            role = m["speaker"]
            orig = m["original"]
            trans = m["translation"] or ""
            convo_text += f"{role.upper()}: {orig}\nEN/HIN: {trans}\n\n"

        prompt = f"""
You are a clinical documentation assistant.
Summarize the following bilingual doctor-patient conversation into a concise,
de-identified clinical note capturing:

- Presenting complaints and duration
- Relevant medical context (if mentioned)
- Any red-flag symptoms or follow-up questions
- Next recommended steps (if implied)

Conversation:
{convo_text}

Now provide the summary in clear English:
""".strip()

        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You create concise, de-identified clinical summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=350,
        )

        summary = resp.choices[0].message.content.strip()
        self.sessions.save_summary(session_id, summary)
        print(f"📝 Saved summary for session {session_id}")
        return summary

    # -------------------------------------------------
    def end_session(self, session_id: str):
        """End the current session — placeholder for UI control."""
        if session_id:
            print(f"🧾 Session {session_id} closed by user.")
        else:
            print("⚠️ No active session to close.")
        return True
