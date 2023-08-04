from rouge import Rouge
from bert_score import score
from nltk.translate.meteor_score import single_meteor_score

def compute_average_meteor_score(candidate, reference):
    """
    Compute the average METEOR score for a set of translations.

    Args:
    inputs (list): List of source sentences.
    reference (list): List of reference target sentences.
    candidate (list): List of candidate target sentences by the model.

    Returns:
    float: The average METEOR score.
    """
    meteor_scores = []

    for output, reference in zip(candidate, reference):
        meteor_score = single_meteor_score(reference.split(), output.split())
        meteor_scores.append(meteor_score)

    average_meteor_score = sum(meteor_scores) / len(meteor_scores)
    return average_meteor_score


def compute_normalized_rouge_score(candidate, reference):
    """
    Compute the normalized ROUGE score for a summarization task.

    Args:
    candidate (str): The summary candidate by the model.
    reference (str): The reference summary.

    Returns:
    float: The normalized ROUGE score.
    """
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)

    avg_f1_score = sum(score['rouge-1']['f'] + score['rouge-2']['f'] + score['rouge-l']['f'] for score in scores) / 3
    normalized_score = avg_f1_score / 1.0
    return normalized_score


def compute_bertscore(candidate, reference):
    """
    Compute the BERTScore for a rephrasing task.

    Args:
    candidate (str): The sentence candidate by the model.
    reference (str): The expected sentence.

    Returns:
    float: The BERTScore.
    """
    _, _, bert_score = score([candidate], [reference], lang="en", model_type="roberta-base", num_layers=4)
    return bert_score.item()

