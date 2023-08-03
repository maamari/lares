# Lares

LARES (vaLidation, evAluation and REliability Solutions) is a Python package designed to assist with the evaluation, validation, and generation of text in various tasks such as translation, summarization, and rephrasing. 

## Features

- **Quantitative and Qualitative Evaluation**: LARES provides options both qualitative and quantitative approaches to evaluating models. Quantitative metrics include METEOR scores for translations, normalized ROUGE scores for summarizations, and BERT scores for rephrasing tasks. Qualitative metrics are computed both from binary user judgements as well as sentiment analysis done on user feedback.

- **Toxicity Identification**: Predicts the toxicity of a given text using a pre-trained toxicity model and iteratively rephrases model responses until below a user-specified toxicity threshold.

## Installation

Requires Python 3.6 or later. You can install using pip via:

```bash
pip install lares
```

## Usage

Here is a basic usage example for translation task:

```python
import openai
from datasets import load_dataset
from lares import *

openai.api_key = '' # replace with your OpenAI API key
dataset = load_dataset("opus100", "en-fr")

for data in dataset["validation"]['translation'][100:110]:
    prompt = data["en"]
    reference = data["fr"]

    input_prompt = "Translate the following to french: "+prompt
    print(input_prompt)
    result = generate(input_prompt, reference, task_type='Translation')

    print(f"Prompt: {prompt}")
    print(f"Reference: {reference}")
    print(f"Generated Response: {result}\n")
```

## Dependencies

- openai==0.27.8
- nltk==3.7
- torch==2.0.1
- transformers==4.31.0
- rouge==1.0.1
- bert_score==0.3.12
- datasets==1.11.0

To be explicit, you can install via:

```bash
pip install openai==0.27.8 nltk==3.7 torch==2.0.1 transformers==4.31.0 rouge==1.0.1 bert_score==0.3.12 datasets==1.11.0
```
