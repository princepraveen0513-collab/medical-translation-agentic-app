import re
import spacy
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# -------------------------------------------------
# Load spaCy English model (for PERSON / basic NER)
# -------------------------------------------------
try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run(
        ["python", "-m", "spacy", "download", "en_core_web_sm"],
        check=False
    )
    nlp_en = spacy.load("en_core_web_sm")


# -------------------------------------------------
# Data classes
# -------------------------------------------------
@dataclass
class PIIEntity:
    type: str           # "name", "age", "phone", "email", "location"
    value: str          # original span
    placeholder: str    # e.g. [NAME_1]
    start: int          # start index in original text
    end: int            # end index in original text


@dataclass
class DeidentificationResult:
    original_text: str
    deidentified_text: str
    entities: List[PIIEntity]

    def to_dict(self):
        return {
            "original_text": self.original_text,
            "deidentified_text": self.deidentified_text,
            "entities": [asdict(e) for e in self.entities],
        }


# -------------------------------------------------
# PII Anonymizer
# -------------------------------------------------
class PIIAnonymizer:
    """
    Bilingual (Hindi + English) PII anonymizer.

    Design choices:
    - Only mark AGE when explicitly expressed as age
      (e.g., "I am 26 years old", "my age is 30", "उम्र 25 साल", "26 साल का हूँ").
    - Phone numbers: 10+ digits with phone-like pattern.
    - Names:
        * English: spaCy PERSON
        * Hindi: "मेरा नाम X है" style patterns
    - Emails: standard pattern.
    - Everything runs locally (no external calls).
    """

    def __init__(self):
        # Phone: flexible formatting, but must contain at least 10 digits
        self.phone_pattern = re.compile(r"(\+?\d[\d\-\s]{8,}\d)")

        # Email: standard-ish
        self.email_pattern = re.compile(
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
        )

        # English age expressions: strong cues only
        self.en_age_patterns = [
            re.compile(
                r"\b(?:i\s*am|i'm|my\s+age\s+is|age\s+is|aged)\s+(\d{1,2})\s*(?:years?\s*old)?\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(\d{1,2})\s*(?:years?\s*old)\b",
                re.IGNORECASE,
            ),
        ]

        # Hindi age expressions: must mention साल/saal with age context
        self.hi_age_patterns = [
            re.compile(
                r"\bउम्र\s*(\d{1,2})\s*(साल|saal)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(\d{1,2})\s*(साल|saal)\s*(का|की|के)\s*(?:हूँ|hai|है)\b",
                re.IGNORECASE,
            ),
        ]

        # Hindi name patterns like "मेरा नाम X है"
        # Capture full name until "है" or punctuation.
        self.hi_name_patterns = [
            re.compile(
                r"मेरा\s+नाम\s+(.+?)(?=\s+है|[।.!?,]|$)",
                re.IGNORECASE,
            ),
        ]

        # Placeholder counters
        self.counters = {
            "name": 1,
            "age": 1,
            "phone": 1,
            "email": 1,
            "location": 1,
        }

    # -------------------------------------------------
    def deidentify(
        self,
        text: str,
        existing_map: Optional[Dict[str, str]] = None,
    ) -> DeidentificationResult:
        """
        Main entrypoint.
        Returns a DeidentificationResult with:
          - original_text
          - deidentified_text
          - list of PIIEntity
        existing_map (placeholder -> value) can be used to keep stable IDs across turns.
        """
        if existing_map is None:
            existing_map = {}

        entities: List[PIIEntity] = []

        # ---------- 1. English NER for PERSON ----------
        doc = nlp_en(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                placeholder = self._get_or_create_placeholder(
                    "name", ent.text, existing_map
                )
                entities.append(
                    PIIEntity(
                        type="name",
                        value=ent.text,
                        placeholder=placeholder,
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )

        # ---------- 2. Hindi explicit name patterns ----------
        for pat in self.hi_name_patterns:
            for m in pat.finditer(text):
                name = m.group(1).strip()
                if len(name) > 1:
                    placeholder = self._get_or_create_placeholder(
                        "name", name, existing_map
                    )
                    entities.append(
                        PIIEntity(
                            type="name",
                            value=name,
                            placeholder=placeholder,
                            start=m.start(1),
                            end=m.end(1),
                        )
                    )

        # ---------- 3. Phone numbers ----------
        phone_spans = []
        for m in self.phone_pattern.finditer(text):
            raw = m.group(1).strip()
            digits = re.sub(r"\D", "", raw)
            if len(digits) >= 10:
                placeholder = self._get_or_create_placeholder(
                    "phone", raw, existing_map
                )
                start, end = m.start(1), m.end(1)
                phone_spans.append((start, end))
                entities.append(
                    PIIEntity(
                        type="phone",
                        value=raw,
                        placeholder=placeholder,
                        start=start,
                        end=end,
                    )
                )

        # ---------- 4. Emails ----------
        for m in self.email_pattern.finditer(text):
            email = m.group(0)
            placeholder = self._get_or_create_placeholder(
                "email", email, existing_map
            )
            entities.append(
                PIIEntity(
                    type="email",
                    value=email,
                    placeholder=placeholder,
                    start=m.start(),
                    end=m.end(),
                )
            )

        # ---------- 5. Ages (with strong context only) ----------
        # English age mentions
        for pat in self.en_age_patterns:
            for m in pat.finditer(text):
                num = m.group(1)
                if not num:
                    continue
                if not (0 < int(num) < 120):
                    continue
                start, end = m.start(1), m.end(1)
                if self._inside_spans(start, phone_spans):
                    continue  # don't mis-tag inside phone numbers
                placeholder = self._get_or_create_placeholder(
                    "age", num, existing_map
                )
                entities.append(
                    PIIEntity(
                        type="age",
                        value=num,
                        placeholder=placeholder,
                        start=start,
                        end=end,
                    )
                )

        # Hindi age mentions
        for pat in self.hi_age_patterns:
            for m in pat.finditer(text):
                num = m.group(1)
                if not num:
                    continue
                if not (0 < int(num) < 120):
                    continue
                start, end = m.start(1), m.end(1)
                if self._inside_spans(start, phone_spans):
                    continue
                placeholder = self._get_or_create_placeholder(
                    "age", num, existing_map
                )
                entities.append(
                    PIIEntity(
                        type="age",
                        value=num,
                        placeholder=placeholder,
                        start=start,
                        end=end,
                    )
                )

        # ---------- 6. Clean up + apply replacements ----------
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda e: e.start)

        deidentified_text = self._apply_replacements(text, entities)

        return DeidentificationResult(
            original_text=text,
            deidentified_text=deidentified_text,
            entities=entities,
        )

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _inside_spans(self, idx: int, spans: List[tuple]) -> bool:
        for s, e in spans:
            if s <= idx < e:
                return True
        return False

    def _get_or_create_placeholder(
        self,
        ent_type: str,
        value: str,
        existing_map: Dict[str, str],
    ) -> str:
        # Reuse existing placeholder if same value already mapped
        for ph, val in existing_map.items():
            if val == value and ph.startswith(f"[{ent_type.upper()}_"):
                return ph

        # Otherwise, create new
        ph = f"[{ent_type.upper()}_{self.counters[ent_type]}]"
        self.counters[ent_type] += 1
        # Note: you can choose to persist this mapping externally if needed
        return ph

    def _deduplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        unique = []
        seen = set()
        for e in entities:
            key = (e.start, e.end, e.type, e.placeholder)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def _apply_replacements(self, text: str, entities: List[PIIEntity]) -> str:
        """Apply replacements left-to-right with offset tracking."""
        result = text
        offset = 0
        for e in entities:
            s = e.start + offset
            t = e.end + offset
            result = result[:s] + e.placeholder + result[t:]
            offset += len(e.placeholder) - (e.end - e.start)
        return result
