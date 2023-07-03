# pii_masker

This repo is forked from https://huggingface.co/spaces/beki/pii-anonymizer/tree/main

### Build the PII data anonymization solution

#### Recognizer

pre-build model or pattern recognizer 

#### Highlighter

Place holder for the highlight solution (now we are using annotated_text)
if we want to do a annotation on HTML, we may need other solution

#### Masker

Replace the PII with labels for now. we can implement other solution to replace it with * or emoji. 


#### Usage of the commandline tool

```
Usage: cli.py [OPTIONS]

  Process the input file and generate the output file.

Options:
  -i, --input_file PATH           Input file path
  -o, --output_file PATH          Output file path
  -m, --model [spacy|flair|pattern|whole]
                                  Model type (predefined_model_with spacy or
                                  flair, pattern_recognizer or combine those
                                  toghether)

  -s, --stats_report              Generate stats report
  --help                          Show this message and exit.
  ```

  #### Demo of the tool

  This tool will process a text file. Generate a annotated html file with PII labels, a text file with Pii data anomymized and a json report with the desicion making progress

