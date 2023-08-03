import evaluate
import validate

def generate(prompt, reference, max_iterations=10, task_type = None):
    """
    Generate a response, evaluate it, and ensure it is safe.

    Args:
    prompt (str): The input prompt for the model.
    reference (str): The reference solution.
    max_iterations (int): The maximum number of iterations.

    Returns:
    str: The suitable response.
    """
    for i in range(max_iterations):
        # Determine task type
        if task_type==None: 
            for _ in range(10):
                task_type = determine_task(prompt)
                if task_type in ["Translation", "Summarization", "Rephrasing"]:
                    break
            if task_type not in ["Translation", "Summarization", "Rephrasing"]:
                raise ValueError(f"Unknown task type: {task_type}")

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
        if score >= 0.3 and toxicity <= 0.7:
            return candidate

        # If the response is not acceptable due to low score, generate a new response
        elif score < 0.3:
#             print(score)
            continue

        # If the response is not acceptable due to high toxicity, generate a safe response
        else:
            candidate = until_safe(candidate)

            # If no safe response is found after max_attempts, raise an exception
            if candidate is None:
                raise ValueError("Failed to generate an acceptable response after max_iterations")

    return None
