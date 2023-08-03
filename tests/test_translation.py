import openai
from datasets import load_dataset
from lares import *

openai.api_key = '' 
dataset = load_dataset("opus100", "en-fr")

for data in dataset["validation"]['translation'][100:110]:
    prompt = data["en"]
    reference = data["fr"]

    # Your function here
    input_prompt = "Translate the following to french: "+prompt
    print(input_prompt)
    result = generate(input_prompt, reference, task_type='Translation')

    print(f"Prompt: {prompt}")
    print(f"Reference: {reference}")
    print(f"Generated Response: {result}\n")

