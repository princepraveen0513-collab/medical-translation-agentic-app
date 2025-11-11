# core/agents/intent_classifier.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class IntentClassifier:
    """
    Lightweight LLM-based classifier that decides whether
    medical RAG retrieval is needed for a given message.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model_name

    def classify_intent(self, message: str) -> dict:
        """
        Returns a structured dict:
        {
            "label": "medical_required" | "not_required" | "small_talk",
            "confidence": float
        }
        """
        prompt = f"""
You are a simple intent classifier for a bilingual medical assistant.

Given a short user message, determine if MEDICAL CONTEXT retrieval
(i.e., running a medical RAG search) is required.

Classify into one of the following:
- "medical_required": if the message contains any symptom, body part, illness, or medication.
- "not_required": if it's logistical, greeting, or non-medical ("hello", "thanks", "good morning").
- "small_talk": if it's friendly conversation or unrelated to health.

Respond strictly in JSON as:
{{"label": "<one_of_above>", "confidence": <0_to_1_float>}}

Message:
\"{message}\"
""".strip()

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You output a strict JSON classification."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=60,
            )
            raw = resp.choices[0].message.content.strip()
            print(f"🧭 IntentClassifier raw output: {raw}")
            import json
            parsed = json.loads(raw)
            label = parsed.get("label", "medical_required")
            conf = float(parsed.get("confidence", 0.7))
        except Exception as e:
            print("⚠️ Intent classification failed:", e)
            label, conf = "medical_required", 0.5

        print(f"🏷️ IntentClassifier → {label.upper()} (conf {conf})")
        return {"label": label, "confidence": conf, "timestamp": datetime.utcnow().isoformat()}
