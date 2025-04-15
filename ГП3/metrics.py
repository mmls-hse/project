from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
import torch
import torch.nn.functional as F

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

def calculate_rouge(reference, candidate, scorer):
    # Вычисление ROUGE
    scores = scorer.score(reference, candidate)
    return scores

def calculate_bleu(reference, candidate):
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    return sentence_bleu(reference_tokens, candidate_tokens)


def calculate_meteor(reference, candidate):
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    return meteor_score([reference_tokens], candidate_tokens)

def calculate_perplexity(model, tokenizer, text, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    model.to(device)

    tokens = tokenizer.encode(text).ids
    input_ids = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        outputs = model(input_ids[:, :-1], input_ids[:, 1:])
        logits = outputs
        log_probs = F.log_softmax(logits, dim=-1)

    nll = 0
    for i in range(len(tokens) - 1):
        nll += -log_probs[0, i, tokens[i + 1]].item()

    perplexity = torch.exp(torch.tensor(nll / (len(tokens) - 1)))
    return perplexity.item()