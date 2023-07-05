import spacy
from recognizer.spacy_recognizer import CustomSpacyRecognizer
from recognizer.spacy_pattern_recognizer import PatternRecognizerFactory
from recognizer.flair_recognizer import FlairRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.pattern import Pattern
import pandas as pd
from annotated_text.util import get_annotated_html
from json import JSONEncoder
import json
import warnings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Helper methods
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
        tokens.append((text[res.start : res.end], res.entity_type))

        # if another entity coming i.e. we're not at the last results element, add text up to next entity
        if i != len(results) - 1:
            tokens.append(text[res.end : results[i + 1].start])
        # if no more entities coming, add all remaining text
        else:
            tokens.append(text[res.end :])
    return tokens


class CustomEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def process_data(input_file, output_file, model, stats_report):
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
    backslash_char = "\\"
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
