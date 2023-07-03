
import logging
from typing import Optional, List, Tuple, Set
from presidio_analyzer import Pattern, PatternRecognizer
#from presidio_analyzer import (
#    RecognizerResult,
#    LocalRecognizer,
#    AnalysisExplanation,
#)
#from presidio_analyzer.nlp_engine import NlpArtifacts
#from presidio_analyzer.predefined_recognizers.spacy_recognizer import SpacyRecognizer
#import re
#import spacy
#import faker



logger = logging.getLogger("presidio-analyzer")


# Initialize the Faker library
fake = faker.Faker()

# Initialize SpaCy
nlp = spacy.load('en_core_web_sm')

patterns = {
    "credit_card": (r'\b(?:\d[ -]*?){13,16}\b', "'**** **** **** ****'"),
    "ssn": (r'\b\d{3}-?\d{2}-?\d{4}\b', '***-**-****'),
    "phone": (r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', "(***-***-****)"),
    "email": (r'\S+@\S+', '****@****.com')
}

def highlight(text):
    """Returns the replacement text, highlighted in red."""
    return f'<span style="background-color: red">{text}</span>'


# Update the anonymization functions to return highlighted replacements.
def anonymize_pattern(text, html, regex, hidden):
    matches = re.findall(regex, text)
    for match in matches:
        highlighted = highlight(match)
        text = text.replace(match, hidden)
        html = html.replace(match, highlighted)
    return text, html, len(matches)


# Define function for anonymizing named entities
def anonymize_named_entities(text, html):
    doc = nlp(text)
    matches = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']]
    for match in matches:
        for match in matches:
            hidden = fake.name()
            highlighted = highlight(match)
            text = text.replace(match, hidden)
            html = html.replace(match, highlighted)
    return text, html, len(matches)


def anonymize_text(text):
    doc = nlp(text)
    #x = next(doc.sents)
    sentences = [sent.text.strip() for sent in doc.sents]

    anonymized_sentences = []
    marked_sentences = []
    report = {key:0 for key in patterns.keys()}
    report["entities"] = 0

    for sentence in sentences:
        original_sentence = html = sentence
        for key, pattern in patterns.items():
            regex, hidden = pattern
            sentence, html, count = anonymize_pattern(sentence, html, regex, hidden)
            report[key] += count

        sentence, html, count = anonymize_named_entities(sentence, html)
        report["entities"] += count

        anonymized_sentences.append(sentence)
        if original_sentence != sentence:  # If the sentence was modified
            marked_sentences.append(html)

    # Join the anonymized sentences into a single string, separated by <br> for line breaks
    marked_text = '<br>'.join(marked_sentences)
    anonymized_text = '\n'.join(anonymized_sentences)

    return anonymized_text, marked_text, report


# Test the function
text = """
John Doe's email is john.doe@example.com.
His credit card number is 1234 5678 9123 4567.
His SSN is 123-45-6789.
His phone number is (123) 456-7890.
"""
anonymized_text, marked_text, report = anonymize_text(text)

print(anonymized_text)
with open("report.html", "w") as fp:
    fp.write(marked_text)
print(report)



import logging
from typing import Optional, List, Tuple, Set

from presidio_analyzer import (
    RecognizerResult,
    EntityRecognizer,
    AnalysisExplanation,
)
from presidio_analyzer.nlp_engine import NlpArtifacts

try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
except ImportError:
    print("Flair is not installed")


logger = logging.getLogger("presidio-analyzer")


class FlairRecognizer(EntityRecognizer):
    """
    Wrapper for a flair model, if needed to be used within Presidio Analyzer.
    :example:
    >from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    >flair_recognizer = FlairRecognizer()
    >registry = RecognizerRegistry()
    >registry.add_recognizer(flair_recognizer)
    >analyzer = AnalyzerEngine(registry=registry)
    >results = analyzer.analyze(
    >    "My name is Christopher and I live in Irbid.",
    >    language="en",
    >    return_decision_process=True,
    >)
    >for result in results:
    >    print(result)
    >    print(result.analysis_explanation)
    """

    ENTITIES = [
        "LOCATION",
        "PERSON",
        "NRP",
        "GPE",
        "ORGANIZATION",
        "MAC_ADDRESS",
        "US_BANK_NUMBER",
        "IMEI",
        "TITLE",
        "LICENSE_PLATE",
        "US_PASSPORT",
        "CURRENCY",
        "ROUTING_NUMBER",
        "US_ITIN",
        "US_BANK_NUMBER",
        "US_DRIVER_LICENSE",
        "AGE",
        "PASSWORD",
        "SWIFT_CODE",
    ]

    DEFAULT_EXPLANATION = "Identified as {} by Flair's Named Entity Recognition"

    CHECK_LABEL_GROUPS = [
        ({"LOCATION"}, {"LOC", "LOCATION", "STREET_ADDRESS", "COORDINATE"}),
        ({"PERSON"}, {"PER", "PERSON"}),
        ({"NRP"}, {"NORP", "NRP"}),
        ({"GPE"}, {"GPE"}),
        ({"ORGANIZATION"}, {"ORG"}),
        ({"MAC_ADDRESS"}, {"MAC_ADDRESS"}),
        ({"US_BANK_NUMBER"}, {"US_BANK_NUMBER"}),
        ({"IMEI"}, {"IMEI"}),
        ({"TITLE"}, {"TITLE"}),
        ({"LICENSE_PLATE"}, {"LICENSE_PLATE"}),
        ({"US_PASSPORT"}, {"US_PASSPORT"}),
        ({"CURRENCY"}, {"CURRENCY"}),
        ({"ROUTING_NUMBER"}, {"ROUTING_NUMBER"}),
        ({"AGE"}, {"AGE"}),
        ({"CURRENCY"}, {"CURRENCY"}),
        ({"SWIFT_CODE"}, {"SWIFT_CODE"}),
        ({"US_ITIN"}, {"US_ITIN"}),
        ({"US_BANK_NUMBER"}, {"US_BANK_NUMBER"}),
        ({"US_DRIVER_LICENSE"}, {"US_DRIVER_LICENSE"}),
    ]

    MODEL_LANGUAGES = {
        "en":"beki/flair-pii-english-large",
        # "en":"flair-trf.pt",
    }

    PRESIDIO_EQUIVALENCES = {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        "NROP": "NRP",
        "URL": "URL",
        "US_ITIN": "US_ITIN",
        "US_PASSPORT": "US_PASSPORT",
        "IBAN_CODE": "IBAN_CODE",
        "IP_ADDRESS": "IP_ADDRESS",
        "EMAIL_ADDRESS": "EMAIL",
        "US_DRIVER_LICENSE": "US_DRIVER_LICENSE",
        "US_BANK_NUMBER": "US_BANK_NUMBER",
    }

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
        check_label_groups: Optional[Tuple[Set, Set]] = None,
        model: SequenceTagger = None,
    ):
        self.check_label_groups = (
            check_label_groups if check_label_groups else self.CHECK_LABEL_GROUPS
        )

        supported_entities = supported_entities if supported_entities else self.ENTITIES
        self.model = (
            model
            if model
            else SequenceTagger.load(self.MODEL_LANGUAGES.get(supported_language))
        )

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="Flair Analytics",
        )

    def load(self) -> None:
        """Load the model, not used. Model is loaded during initialization."""
        pass

    def get_supported_entities(self) -> List[str]:
        """
        Return supported entities by this model.
        :return: List of the supported entities.
        """
        return self.supported_entities

    # Class to use Flair with Presidio as an external recognizer.
    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None
    ) -> List[RecognizerResult]:
        """
        Analyze text using Text Analytics.
        :param text: The text for analysis.
        :param entities: Not working properly for this recognizer.
        :param nlp_artifacts: Not used by this recognizer.
        :param language: Text language. Supported languages in MODEL_LANGUAGES
        :return: The list of Presidio RecognizerResult constructed from the recognized
            Flair detections.
        """

        results = []

        sentences = Sentence(text)
        self.model.predict(sentences)

        # If there are no specific list of entities, we will look for all of it.
        if not entities:
            entities = self.supported_entities

        for entity in entities:
            if entity not in self.supported_entities:
                continue

            for ent in sentences.get_spans("ner"):
                if not self.__check_label(
                    entity, ent.labels[0].value, self.check_label_groups
                ):
                    continue
                textual_explanation = self.DEFAULT_EXPLANATION.format(
                    ent.labels[0].value
                )
                explanation = self.build_flair_explanation(
                    round(ent.score, 2), textual_explanation
                )
                flair_result = self._convert_to_recognizer_result(ent, explanation)

                results.append(flair_result)

        return results

    def _convert_to_recognizer_result(self, entity, explanation) -> RecognizerResult:

        entity_type = self.PRESIDIO_EQUIVALENCES.get(entity.tag, entity.tag)
        flair_score = round(entity.score, 2)

        flair_results = RecognizerResult(
            entity_type=entity_type,
            start=entity.start_position,
            end=entity.end_position,
            score=flair_score,
            analysis_explanation=explanation,
        )

        return flair_results

    def build_flair_explanation(
        self, original_score: float, explanation: str
    ) -> AnalysisExplanation:
        """
        Create explanation for why this result was detected.
        :param original_score: Score given by this recognizer
        :param explanation: Explanation string
        :return:
        """
        explanation = AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    @staticmethod
    def __check_label(
        entity: str, label: str, check_label_groups: Tuple[Set, Set]
    ) -> bool:
        return any(
            [entity in egrp and label in lgrp for egrp, lgrp in check_label_groups]
        )


if __name__ == "__main__":

    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry

    flair_recognizer = (
        FlairRecognizer()
    )  # This would download a very large (+2GB) model on the first run

    registry = RecognizerRegistry()
    registry.add_recognizer(flair_recognizer)

    analyzer = AnalyzerEngine(registry=registry)

    results = analyzer.analyze(
        "{first_name: Moustafa, sale_id: 235234}",
        language="en",
        return_decision_process=True,
    )
    for result in results:
        print(result)
        print(result.analysis_explanation)


import logging
from typing import Optional, List, Tuple, Set

from presidio_analyzer import (
    RecognizerResult,
    LocalRecognizer,
    AnalysisExplanation,
)
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.predefined_recognizers.spacy_recognizer import SpacyRecognizer

logger = logging.getLogger("presidio-analyzer")


class CustomSpacyRecognizer(LocalRecognizer):

    ENTITIES = [
        "LOCATION",
        "PERSON",
        "NRP",
        "ORGANIZATION",
        "DATE_TIME",
    ]

    DEFAULT_EXPLANATION = "Identified as {} by Spacy's Named Entity Recognition (Privy-trained)"

    CHECK_LABEL_GROUPS = [
        ({"LOCATION"}, {"LOC", "LOCATION", "STREET_ADDRESS", "COORDINATE"}),
        ({"PERSON"}, {"PER", "PERSON"}),
        ({"NRP"}, {"NORP", "NRP"}),
        ({"ORGANIZATION"}, {"ORG"}),
        ({"DATE_TIME"}, {"DATE_TIME"}),
    ]

    MODEL_LANGUAGES = {
        "en": "beki/en_spacy_pii_distilbert",
    }

    PRESIDIO_EQUIVALENCES = {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        "NROP": "NRP",
        "DATE_TIME": "DATE_TIME",
    }

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
        check_label_groups: Optional[Tuple[Set, Set]] = None,
        context: Optional[List[str]] = None,
        ner_strength: float = 0.85,
    ):
        self.ner_strength = ner_strength
        self.check_label_groups = (
            check_label_groups if check_label_groups else self.CHECK_LABEL_GROUPS
        )
        supported_entities = supported_entities if supported_entities else self.ENTITIES
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
        )

    def load(self) -> None:
        """Load the model, not used. Model is loaded during initialization."""
        pass

    def get_supported_entities(self) -> List[str]:
        """
        Return supported entities by this model.
        :return: List of the supported entities.
        """
        return self.supported_entities

    def build_spacy_explanation(
        self, original_score: float, explanation: str
    ) -> AnalysisExplanation:
        """
        Create explanation for why this result was detected.
        :param original_score: Score given by this recognizer
        :param explanation: Explanation string
        :return:
        """
        explanation = AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    def analyze(self, text, entities, nlp_artifacts=None):  # noqa D102
        results = []
        if not nlp_artifacts:
            logger.warning("Skipping SpaCy, nlp artifacts not provided...")
            return results

        ner_entities = nlp_artifacts.entities

        for entity in entities:
            if entity not in self.supported_entities:
                continue
            for ent in ner_entities:
                if not self.__check_label(entity, ent.label_, self.check_label_groups):
                    continue
                textual_explanation = self.DEFAULT_EXPLANATION.format(
                    ent.label_)
                explanation = self.build_spacy_explanation(
                    self.ner_strength, textual_explanation
                )
                spacy_result = RecognizerResult(
                    entity_type=entity,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=self.ner_strength,
                    analysis_explanation=explanation,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name
                    },
                )
                results.append(spacy_result)

        return results

    @staticmethod
    def __check_label(
        entity: str, label: str, check_label_groups: Tuple[Set, Set]
    ) -> bool:
        return any(
            [entity in egrp and label in lgrp for egrp, lgrp in check_label_groups]
        )


