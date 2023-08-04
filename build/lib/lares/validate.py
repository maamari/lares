import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from safetensors import safe_open
from fairlearn.metrics import MetricFrame
import numpy as np

def compute_toxicity(candidate):
    """
    Predict the toxicity of some text.

    Args:
    candidate (str): The text to evaluate.

    Returns:
    float: The toxicity score.
    """
    toxicity_model_name = "cardiffnlp/twitter-roberta-base-offensive"
    toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name, model_max_length=1000)
    toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name)

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
    base_model_name = 'facebook/bart-base'
    model_name = 'SkolkovoInstitute/bart-base-detox'
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, model_max_length=1000)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_safe_str = candidate
    for _ in range(max_attempts):
        inputs = tokenizer([current_safe_str], return_tensors="pt", truncation=True)
        inputs = inputs.to(device)
        paraphrase = model.generate(**inputs)
        safe_str = tokenizer.decode(paraphrase[0], skip_special_tokens=True)

        toxicity_score = compute_toxicity(safe_str)
        similarity_score = compute_bertscore(candidate, safe_str)
        print(candidate, safe_str, toxicity_score, similarity_score)

        if (toxicity_score < target_toxicity) and (similarity_score > target_similarity):
            return safe_str, toxicity_score, similarity_score

        current_safe_str = safe_str

    return None, toxicity_score, similarity_score


def calculate_bias(data):
    """
    Calculate the bias score based on translation scores.

    Args:
    data (dict): The dictionary containing responses, references, labels, toxicity, and scores.

    Returns:
    dict, float: The scores for each group and the bias score.
    """
    # Define the sensitive features
    sensitive_features = np.array(['fr' if label == 0 else 'es' for label in data['label']])

    # Define the function to calculate the mean score
    def mean_score(y_true, y_pred):
        return np.mean(y_pred)

    # Compute the fairness metrics for the scores
    scores_metric_frame = MetricFrame(
        metrics=mean_score,
        y_true=data['reference'],
        y_pred=data['score'],
        sensitive_features=sensitive_features
    )

    # Get the mean scores for each group
    group_scores = scores_metric_frame.by_group

    # Compute the absolute difference between the group scores
    diff = np.abs(group_scores['fr'] - group_scores['es'])

    # Normalize the difference to get a bias score between 0 and 1
    bias_score = diff

    return group_scores, bias_score

