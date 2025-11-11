import os
import sys

# Allow importing from project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from core.pii.pii_agent import PIIAnonymizer

if __name__ == "__main__":
    anonymizer = PIIAnonymizer()

    #text = "मेरा नाम प्रिंस प्रवीन है, मैं 26 साल का हूँ और मेरा मोबाइल 9876543210 है।"
    text = "I am 34 years old and have chest pain"
    result = anonymizer.deidentify(text)

    print("Original:")
    print(result.original_text)
    print("\nDeidentified:")
    print(result.deidentified_text)
    print("\nEntities:")
    for e in result.entities:
        print(e)
