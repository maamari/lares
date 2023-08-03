LARES (vaLidation, evAluation and REliability Solutions) is a Python package designed to assist with the evaluation and validation models in various tasks such as translation, summarization, and rephrasing. 

This package leverages a suite of existing tools and resources to provide the best form of evaluation and validation for the prompted task. Natural Language Toolkit (NLTK), BERT, and ROUGE are employed for evaluations, while Microsoft's Fairlearn, Facebook's BART, and roBERTa are used to assess and address the toxicity and fairness of a given model.

In addition, LARES uses datasets from HuggingFace, where the choice of datasets was informed by benchmark setters such as the General Language Understanding Evaluation (GLUE) benchmark.

## Features

- **Quantitative and Qualitative Evaluation**: Provides both qualitative and quantitative approaches to evaluating models. Quantitative metrics include METEOR scores for translations, normalized ROUGE scores for summarizations, and BERT scores for rephrasing tasks. Qualitative metrics are computed both from binary user judgements as well as sentiment analysis done on user feedback.

- **Fairness and Toxicity Validation**: Provides a quantitative measure of the toxicity and fairness of a given model for specific tasks by leveraging Fairlearn and roBERTa. 

- **Iterative Reconstruction**: Iteratively rephrases model responses until below a specified toxicity and above a specified quality threshold using BART 

## Workflow


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
