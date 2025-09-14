# src/llm_normalize.py
from typing import List

SYSTEM = """You rewrite short patient intents into clear, non-ambiguous,
patient-friendly medical phrases suitable for triage. Keep ≤ 8 words.
Examples: 'heart hurt'→'chest pain'; 'sugar low'→'possible hypoglycaemia'."""

def normalize(phrase: str) -> str:
    # Placeholder: no external calls required for demo.
    # Rule-based fallback; you can replace with an LLM API later.
    lex = {
        "heart hurt":"chest pain",
        "pain chest":"chest pain",
        "sugar low":"possible hypoglycaemia",
        "breath difficult":"shortness of breath",
        "need help":"need assistance",
        "feel sick":"nausea",
        "head pain":"headache",
        "dizzy":"dizziness",
        "water":"need water",
        "toilet":"need toilet",
    }
    s = phrase.lower().strip()
    return lex.get(s, phrase)