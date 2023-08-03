def compute_toxicity(candidate):
    """
    Predict the toxicity of some text.

    Args:
    candidate (str): The text to evaluate.

    Returns:
    float: The toxicity score.
    """
    inputs = toxicity_tokenizer(candidate, return_tensors="pt", truncation=True, padding=True)
    outputs = toxicity_model(**inputs)
    probabilities = torch.sigmoid(outputs.logits).detach().numpy()[0]
    toxicity_score = probabilities[1]
    return toxicity_score

def until_safe(candidate, target_toxicity=0.3, target_similarity=0.3, max_attempts=10):
    """
    Generate a paraphrase of the input string that is both semantically similar and not toxic.

    Args:
    candidate (str): The initial input string to paraphrase.
    target_toxicity (float, optional): The maximum acceptable toxicity score. Default is 0.3.
    target_similarity (float, optional): The minimum acceptable similarity score. Default is 0.3.
    max_attempts (int, optional): The maximum number of paraphrase attempts. Default is 10.

    Returns:
    str: The safe paraphrase, or None if no safe paraphrase is found after max_attempts.
    """
    current_safe_str = candidate
    for _ in range(max_attempts):
        inputs = tokenizer([current_safe_str], return_tensors="pt", max_length=100, truncation=True)
        inputs = inputs.to(device)
        paraphrase = model.generate(**inputs)
        safe_str = tokenizer.decode(paraphrase[0], skip_special_tokens=True)
        
        toxicity_score = compute_toxicity(safe_str)
        similarity_score = compute_bertscore(candidate, safe_str)
        print(candidate, safe_str, toxicity_score, similarity_score)

        if (toxicity_score < target_toxicity) and (similarity_score > target_similarity):
            return safe_str

        current_safe_str = safe_str
        
    return None

