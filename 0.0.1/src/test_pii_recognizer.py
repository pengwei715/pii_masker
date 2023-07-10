import os
import pytest
from pii_recognizer import (
    process,
    analyzer_engine,
    anonymize,
    annotate,
    get_supported_entities,
)


@pytest.fixture(params=["whole", "spacy", "flair", "pattern"])
def model(request):
    return request.param

@pytest.fixture(params=["en"])
def language(request):
    return request.param

#test the process function
def test_process(model, language):
    """Test the process function."""
    input_file = "tests/data/test.txt"
    output_file = "tests/data/test_output.txt"
    process(input_file, output_file, model)
    with open(output_file, "r") as f:
        output = f.read()
    assert output == "John Smith was born in 1990 and now lives in New York City."
    os.remove(output_file)
    os.remove(output_file[:-4] + ".html")
    os.remove(f"{output_file[:-4]}_stats.json")

#test the analyze function

def test_analyze(model, language):
    """Test the analyze function."""
    analyzer = analyzer_engine(model)
    with open("tests/data/test.txt", "r") as f:
        text = f.read()
        results = analyzer.analyze(
            text=text,
            language=language,
            entities=get_supported_entities(),
            return_decision_process=True,
        )
        anonymized_text = anonymize(text, results)
    assert anonymized_text == "John Smith was born in 1990 and now lives in New York City."

#test the anonymize function

def test_anonymize(model, language):
    """Test the anonymize function."""
    analyzer = analyzer_engine(model)
    with open("tests/data/test.txt", "r") as f:
        text = f.read()
        results = analyzer.analyze(
            text=text,
            language=language,
            entities=get_supported_entities(),
            return_decision_process=True,
        )
        anonymized_text = anonymize(text, results)
    assert anonymized_text == "John Smith was born in 1990 and now lives in New York City."

#test the annotate function

def test_annotate(model, language):
    """Test the annotate function."""
    analyzer = analyzer_engine(model)
    with open("tests/data/test.txt", "r") as f:
        text = f.read()
        results = analyzer.analyze(
            text=text,
            language=language,
            entities=get_supported_entities(),
            return_decision_process=True,
        )
        tokens = annotate(text, results, get_supported_entities())
    assert tokens == [
        "John Smith was born in ",
        ("1990", "DATE_TIME"),
        " and now lives in ",
        ("New York City", "LOCATION"),
        ".",
    ]

#test the get_supported_entities function

def test_get_supported_entities(model, language):
    """Test the get_supported_entities function."""
    entities = get_supported_entities()
    assert entities == [
        "AGE",
        "DATE_TIME",
        "EMAIL_ADDRESS",
        "IBAN_CODE",
        "IP_ADDRESS",
        "LOCATION",
        "NRP",
        "ORGANIZATION",
        "PERSON",
        "PHONE_NUMBER",
        "STREET_ADDRESS",
        "URL",
    ]
