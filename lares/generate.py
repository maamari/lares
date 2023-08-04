import openai
from tqdm import tqdm
from collections import defaultdict
from .validate import *
from .evaluate import *

def get_response(prompt):
    """
    Get a response to a given prompt using the text-davinci-003 engine.

    Args:
    prompt (str): The input prompt for the model.

    Returns:
    str: The response generated by the model.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )

    return response.choices[0].text.strip()


def determine_task(task_desc):
    """
    Determine the evaluation tool most suitable for a given task description.

    Args:
    task_desc (str): The description of the task.

    Returns:
    str: The evaluation tool (Translation, Summarization, or Rephrasing).
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"The task is: '{task_desc}'. Which evaluation tool is most suitable for this task? Respond with a single word out of the following choices: Translation, Summarization, Rephrasing",
        temperature=0.5,
        max_tokens=3
    )
    return response.choices[0].text.strip()


def generate(prompts, references, labels, max_iterations=10, task_type=None, feedback=True):
    """
    Generate a response, evaluate it, and ensure it is safe.

    Args:
    prompts (list): The input prompts for the model.
    references (list): The reference solutions.
    labels (list): The labels for each prompt-reference pair.
    max_iterations (int): The maximum number of iterations.

    Returns:
    dict, float, dict: The suitable responses, references, labels, toxicity, and scores, 
    the bias score, and the average score and average toxicity for each group.
    """
    # Check if prompts, references, and labels are lists and have the same length
    if not isinstance(prompts, list) or not isinstance(references, list) or not isinstance(labels, list):
        raise TypeError("Prompts, references, and labels should be lists.")
    if len(prompts) != len(references) != len(labels):
        raise ValueError("The number of prompts, references, and labels should be equal.")

    data = {
        'response': [],
        'reference': [],
        'label': [],
        'toxicity': [],
        'score': []
    }

    for prompt, reference, label in tqdm(zip(prompts, references, labels), total=len(prompts)):
        # Determine task type
        if task_type == None:
            for _ in range(10):
                task_type = determine_task(prompt)
                if task_type in ["Translation", "Summarization", "Rephrasing"]:
                    break
            if task_type not in ["Translation", "Summarization", "Rephrasing"]:
                raise ValueError(f"Unknown task type: {task_type}")

        for i in range(max_iterations):
            # Generate a response
            candidate = get_response(prompt)

            # Evaluate the response and calculate toxicity
            if task_type == "Translation":
                score = compute_average_meteor_score([candidate], [reference])
            elif task_type == "Summarization":
                score = compute_normalized_rouge_score(candidate, reference)
            elif task_type == "Rephrasing":
                score = compute_bertscore(candidate, reference)

            toxicity = compute_toxicity(candidate)

            # Check if the response is acceptable
            if score >= 0.4 and toxicity <= 0.7:
                data['response'].append(candidate)
                data['reference'].append(reference)
                data['label'].append(label)
                data['toxicity'].append(toxicity)
                data['score'].append(score)
                break

            # If the response is not acceptable due to low score, generate a new response
            elif score < 0.4:
                if feedback:
                    if i == max_iterations - 1:
                        print(f"Response: {candidate}")
                        print(f"Reference: {reference}")
                        user_decision = input("Do you want to score this response? (yes/no): ")
                        if user_decision.lower() == "yes":
                            user_score = float(input("Please score the response on a scale from 0 to 1: "))
                            score = user_score
                            data['response'].append(candidate)
                            data['reference'].append(reference)
                            data['label'].append(label)
                            data['toxicity'].append(toxicity)
                            data['score'].append(score)
                            break
                continue

            # If the response is not acceptable due to high toxicity, generate a safe response
            else:
                candidate, toxicity, score = until_safe(candidate)

                # If no safe response is found after max_attempts, raise an exception
                if candidate is None:
                    raise ValueError("Failed to generate an acceptable response after max_iterations")
    
    # Calculate bias score
    _, bias_score = calculate_bias(data)
    
    # Calculate average score and toxicity for each group
    group_scores = defaultdict(list)
    group_toxicity = defaultdict(list)
    for label, score, toxicity in zip(data['label'], data['score'], data['toxicity']):
        group_scores[label].append(score)
        group_toxicity[label].append(toxicity)

    avg_group_scores = {label: np.mean(scores) for label, scores in group_scores.items()}
    avg_group_toxicity = {label: np.mean(toxicity) for label, toxicity in group_toxicity.items()}
    
    return data, bias_score, avg_group_scores, avg_group_toxicity

