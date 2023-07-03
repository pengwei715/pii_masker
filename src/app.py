"""Streamlit app for Spacy trained model and pattern recgnizer, flair model"""

import spacy
from recognizer.spacy_recognizer import CustomSpacyRecognizer
from recognizer.spacy_pattern_recognizer import PatternRecognizerFactory
from recognizer.flair_recognizer import FlairRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.pattern import Pattern
import pandas as pd
from annotated_text import annotated_text
from json import JSONEncoder
import json
import warnings
import streamlit as st
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Helper methods
@st.cache_resource()
def analyzer_engine():
    """Return AnalyzerEngine."""

    registry = RecognizerRegistry()
    # add the custom build spacy recognizer
    spacy_recognizer = CustomSpacyRecognizer()
    registry.add_recognizer(spacy_recognizer)

    # add the custom build flair recognizer
    flair_recognizer = FlairRecognizer()
    registry.add_recognizer(flair_recognizer)

    # add the pattern recognizer
    pattern_recognizer_factory = PatternRecognizerFactory()
    for recognizer in pattern_recognizer_factory.create_pattern_recognizer():
        registry.add_recognizer(recognizer)

    analyzer = AnalyzerEngine(registry=registry, supported_languages=["en"])
    return analyzer


@st.cache_resource()
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


st.set_page_config(page_title="Privy + Presidio demo (English)", layout="wide")

# Side bar
st.sidebar.markdown(
    """
Detect and anonymize PII in text using an [NLP model](https://huggingface.co/beki/en_spacy_pii_distilbert) trained on protocol traces (JSON, SQL, XML etc.) generated by
[Privy](https://github.com/pixie-io/pixie/tree/main/src/datagen/pii/privy) and rule-based classifiers from [Presidio](https://aka.ms/presidio).
"""
)

st_entities = st.sidebar.multiselect(
    label="Which entities to look for?",
    options=get_supported_entities(),
    default=list(get_supported_entities()),
)

st_threshold = st.sidebar.slider(
    label="Acceptance threshold", min_value=0.0, max_value=1.0, value=0.35
)

st_return_decision_process = st.sidebar.checkbox("Add analysis explanations in json")

st.sidebar.info(
    "Privy is an open source framework for synthetic data generation in protocol trace formats (json, sql, html etc). Presidio is an open source framework for PII detection and anonymization. "
    "For more info visit [privy](https://github.com/pixie-io/pixie/tree/main/src/datagen/pii/privy) and [aka.ms/presidio](https://aka.ms/presidio)"
)


# Main panel
analyzer_load_state = st.info(
    "Starting Presidio analyzer and loading Privy-trained PII model..."
)
engine = analyzer_engine()
analyzer_load_state.empty()


st_text = st.text_area(
    label="Type in some text",
    value="SELECT shipping FROM users WHERE shipping = '201 Thayer St Providence RI 02912'"
    "\n\n"
    "{user: Willie Porter, ip: 192.168.2.80, email: willie@gmail.com}",
    height=200,
)

button = st.button("Detect PII")

if "first_load" not in st.session_state:
    st.session_state["first_load"] = True

# After
st.subheader("Analyzed")
with st.spinner("Analyzing..."):
    if button or st.session_state.first_load:
        st_analyze_results = analyze(
            text=st_text,
            entities=st_entities,
            language="en",
            score_threshold=st_threshold,
            return_decision_process=st_return_decision_process,
        )
        annotated_tokens = annotate(st_text, st_analyze_results, st_entities)
        # annotated_tokens
        annotated_text(*annotated_tokens)
# vertical space
st.text("")

st.subheader("Anonymized")

with st.spinner("Anonymizing..."):
    if button or st.session_state.first_load:
        st_anonymize_results = anonymize(st_text, st_analyze_results)
        st_anonymize_results


# table result
st.subheader("Detailed Findings")
if st_analyze_results:
    res_dicts = [r.to_dict() for r in st_analyze_results]
    for d in res_dicts:
        d["Value"] = st_text[d["start"] : d["end"]]
    df = pd.DataFrame.from_records(res_dicts)
    df = df[["entity_type", "Value", "score", "start", "end"]].rename(
        {
            "entity_type": "Entity type",
            "start": "Start",
            "end": "End",
            "score": "Confidence",
        },
        axis=1,
    )

    st.dataframe(df, width=1000)
else:
    st.text("No findings")

st.session_state["first_load"] = True

# json result


class ToDictListEncoder(JSONEncoder):
    """Encode dict to json."""

    def default(self, o):
        """Encode to JSON using to_dict."""
        if o:
            return o.to_dict()
        return []


if st_return_decision_process:
    st.json(json.dumps(st_analyze_results, cls=ToDictListEncoder))