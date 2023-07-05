from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from annotated_text.util import get_annotated_html
from json import JSONEncoder
import json
import warnings
import os
import mlrun

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Helper methods

from presidio_analyzer import PatternRecognizer
from presidio_analyzer.pattern import Pattern

ENTITIES = {
    "CREDIT_CARD": [Pattern("CREDIT_CARD", r"\b(?:\d[ -]*?){13,16}\b", 0.5)],
    "SSN": [Pattern("SSN", r"\b\d{3}-?\d{2}-?\d{4}\b", 0.5)],
    "PHONE": [Pattern("PHONE", r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", 0.5)],
    "EMAIL": [Pattern("EMAIL", r"\S+@\S+", 0.5)],
}


class PatternRecognizerFactory:
    @staticmethod
    def create_pattern_recognizer():
        res = []
        for entity, pattern in ENTITIES.items():
            res.append(PatternRecognizer(supported_entity=entity, patterns=pattern))
        return res
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

    DEFAULT_EXPLANATION = (
        "Identified as {} by Spacy's Named Entity Recognition (Privy-trained)"
    )

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
                textual_explanation = self.DEFAULT_EXPLANATION.format(ent.label_)
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
        "en": "beki/flair-pii-distilbert",
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

def analyzer_engine(model="whole"):
    """Return AnalyzerEngine."""
    registry = RecognizerRegistry()

    spacy_recognizer = CustomSpacyRecognizer()

    flair_recognizer = FlairRecognizer()

    if model == "spacy":
        # add the custom build spacy recognizer
        registry.add_recognizer(spacy_recognizer)
    if model == "flair":
        # add the custom build flair recognizer
        registry.add_recognizer(flair_recognizer)
    if model == "pattern":
        # add the pattern recognizer
        pattern_recognizer_factory = PatternRecognizerFactory()
        for recognizer in pattern_recognizer_factory.create_pattern_recognizer():
            registry.add_recognizer(recognizer)
    if model == "whole":
        registry.add_recognizer(spacy_recognizer)
        registry.add_recognizer(flair_recognizer)
        # add the pattern recognizer
        pattern_recognizer_factory = PatternRecognizerFactory()
        for recognizer in pattern_recognizer_factory.create_pattern_recognizer():
            registry.add_recognizer(recognizer)

    analyzer = AnalyzerEngine(registry=registry, supported_languages=["en"])
    return analyzer


def anonymizer_engine():
    """Return AnonymizerEngine."""
    return AnonymizerEngine()


def get_supported_entities():
    """Return supported entities from the Analyzer Engine."""
    return analyzer_engine().get_supported_entities()


def analyze(**kwargs):
    """Analyze input using Analyzer engine and input arguments (kwargs)."""
    if "entities" not in kwargs or "All" in kwargs["entities"]:
        kwargs["entities"] = None
    return analyzer_engine().analyze(**kwargs)


def anonymize(text, analyze_results):
    """Anonymize identified input using Presidio Abonymizer."""
    if not text:
        return
    res = anonymizer_engine().anonymize(text, analyze_results)
    return res.text


def annotate(text, st_analyze_results, st_entities):
    tokens = []
    # sort by start index
    results = sorted(st_analyze_results, key=lambda x: x.start)
    for i, res in enumerate(results):
        if i == 0:
            tokens.append(text[: res.start])

        # append entity text and entity type
        tokens.append((text[res.start: res.end], res.entity_type))

        # if another entity coming i.e. we're not at the last results element,
        # add text up to next entity
        if i != len(results) - 1:
            tokens.append(text[res.end: results[i + 1].start])
        # if no more entities coming, add all remaining text
        else:
            tokens.append(text[res.end:])
    return tokens


class CustomEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

@mlrun.handler(name="process")
def process(input_file, output_file, model, stats_report):
    """Process the input file and generate the output file."""
    analyzer = analyzer_engine(model)
    with open(input_file, "r") as f:
        text = f.read()
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=get_supported_entities(),
            return_decision_process=True,
        )
        anonymized_text = anonymize(text, results)

    with open(output_file, "w") as f:
        f.write(anonymized_text)
    annotated_tokens = annotate(text, results, get_supported_entities())
    html = get_annotated_html(*annotated_tokens)
    with open(output_file[:-4] + ".html", "w") as f:
        f.write(
            f"<html><body><p>{html.replace('{backslash_char}n', '<br>')}</p></body></html>"
        )
    if stats_report:
        stats = results
        with open(f"{output_file[:-4]}_stats.json", "w") as f:
            json.dump(stats, f, cls=CustomEncoder)
