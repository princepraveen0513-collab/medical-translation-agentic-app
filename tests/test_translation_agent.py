import os
import sys

# Dynamically add project root to Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from core.agents.translation_agent import TranslationAgent


if __name__ == "__main__":
    agent = TranslationAgent()

    # Hindi to English
    hi_text = "मुझे सीने में जलन है और खट्टी डकारें आ रही हैं।"
    medical = ["Possible acid reflux or GERD"]
    cultural = ["Common Hindi phrase indicating acidity, not cardiac pain"]
    print("\n🔹 HINDI → ENGLISH:")
    print(agent.translate_with_context(hi_text, medical, cultural))

    # English to Hindi
    en_text = "Please describe where exactly you feel the pain."
    print("\n🔹 ENGLISH → HINDI:")
    print(agent.translate_with_context(en_text, medical, cultural))
