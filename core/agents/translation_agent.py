import os
from dotenv import load_dotenv
from openai import OpenAI
from langdetect import detect
from textwrap import shorten
import re

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
SYSTEM_PROMPT = os.getenv("TRANSLATION_SYSTEM_PROMPT")

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)


class TranslationAgent:
    """
    Bidirectional translation agent for doctor–patient communication.

    - Detects source language (Hindi/English)
    - Translates accordingly using OpenAI LLM
    - Injects contextual cues (medical, cultural, conversational memory)
    - Returns translation + full prompt for audit/logging
    """

    def __init__(self, model_name: str = OPENAI_MODEL):
        self.model = model_name
        self.system_prompt = SYSTEM_PROMPT or (
            "You are a bilingual medical translation assistant."
        )

    # -------------------------------------------------
    def _detect_direction(self, text: str) -> str:
        """Detect language to choose translation direction."""
        try:
            lang = detect(text)
            if lang in ["hi", "ne"]:
                return "hi_to_en"
            elif lang == "en":
                return "en_to_hi"
            else:
                if any("\u0900" <= c <= "\u097F" for c in text):
                    return "hi_to_en"
                return "en_to_hi"
        except Exception:
            return "hi_to_en"

    # -------------------------------------------------
    def _clean_summary(self, summary_text: str) -> str:
        """🧠 Clean stored summaries (remove markdown, bullets, etc.)"""
        if not summary_text:
            return ""

        # Remove Markdown symbols (**bold**, etc.)
        cleaned = re.sub(r"[*_`#>-]+", "", summary_text)
        # Collapse multiple spaces/newlines
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Truncate extremely long summaries if needed
        return cleaned[:1000]

    # -------------------------------------------------
    def _build_prompt(
        self,
        text: str,
        direction: str,
        medical_context,
        cultural_context,
        conversation_summary: str = None
    ):
        """Build a structured translation prompt."""

        med_snippets = "\n".join(
            [f"- {shorten(txt, width=350, placeholder='...')}" for txt in medical_context[:4]]
        )
        cult_snippets = "\n".join(
            [f"- {shorten(txt, width=350, placeholder='...')}" for txt in cultural_context[:3]]
        )

        direction_note = (
            "Translate from Hindi → English (for the doctor)."
            if direction == "hi_to_en"
            else "Translate from English → Hindi (for the patient)."
        )

        # 🧠 Clean memory summary before injecting
        memory_text = ""
        if conversation_summary:
            cleaned_summary = self._clean_summary(conversation_summary)
            memory_text = (
                f"🧠 Prior Conversation Summary (for context only):\n{cleaned_summary}\n"
            )
        else:
            memory_text = "(no prior summary available)"

        prompt = f"""
{direction_note}

{memory_text}

🩺 Current Message:
{text}

📘 Medical Context:
{med_snippets or '(none found)'}

🎭 Cultural Context:
{cult_snippets or '(none found)'}

🧭 Translation Guidance:
- Use the conversation summary to preserve context (age, symptoms, duration, etc.)
- Do NOT do a word-by-word literal translation.
- Use medical and cultural context to interpret the true meaning.
- If idioms or cultural phrases appear, replace them with medically relevant equivalents.
- Keep the tone clear, empathetic, and natural.
- Ensure the translation would make sense to a clinician or patient.
- Never include personal identifiers or unrelated details.

Now provide only the final translated sentence:
"""
        return prompt.strip()

    # -------------------------------------------------
    def translate_with_context(
        self,
        text: str,
        medical_context,
        cultural_context,
        conversation_summary: str = None
    ):
        """Perform context-aware bidirectional translation."""
        direction = self._detect_direction(text)
        prompt = self._build_prompt(
            text,
            direction,
            medical_context,
            cultural_context,
            conversation_summary,
        )

        # 🔍 Enhanced console output — visually confirm memory context
        print(f"\n🧭 Translation direction: {direction}")
        print("==============================================")
        print("🧩 FINAL PROMPT SENT TO LLM:")
        print("==============================================")
        if conversation_summary:
            print("✅ Conversation Memory INCLUDED in context.")
        else:
            print("⚠️ No conversation memory yet (first few turns).")
        print(prompt)
        print("==============================================\n")

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=250,
            )

            translation = response.choices[0].message.content.strip()

            return {
                "translation": translation,
                "prompt": prompt,
                "direction": direction,
            }

        except Exception as e:
            print("⚠️ Translation failed:", str(e))
            return {
                "translation": "(translation error)",
                "prompt": prompt,
                "direction": direction,
            }
