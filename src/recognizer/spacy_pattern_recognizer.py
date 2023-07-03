
from presidio_analyzer import PatternRecognizer
from presidio_analyzer.pattern import Pattern

ENTITIES = {
            "CREDIT_CARD": [Pattern("CREDIT_CARD", r'\b(?:\d[ -]*?){13,16}\b', 0.5)],
            "SSN": [Pattern("SSN", r'\b\d{3}-?\d{2}-?\d{4}\b', 0.5)],
            "PHONE": [Pattern("PHONE", r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', 0.5)],
            "EMAIL": [Pattern("EMAIL", r'\S+@\S+', 0.5)]
            }

class PatternRecognizerFactory:

    @staticmethod
    def create_pattern_recognizer():
        res = []
        for entity, pattern in ENTITIES.items():
            res.append(
                PatternRecognizer(supported_entity=entity, patterns=pattern)
            )
        return res
