import math


def get_judge_probability(
    judge_logprobs: dict[str, float],
    positive_tokens: list[str] = ["YES"],
    negative_tokens: list[str] = ["NO"],
    min_prob: float = 0.25,
) -> float | None:
    """Parse the logprobs into a probability.
    
    Args:
        judge_logprobs (dict[str, float]): Dictionary of tokens to logprobs, e.g. {'YES': -0.1, 'NO': -0.2}.
        positive_token (str): The token to interpret as a positive classification.
        negative_token (str): The token to interpret as a negative classification.
        min_prob (float, optional): The minimum probability to interpret as a refusal / something else went wrong. Defaults to 0.25.

    Return:
        float | None: The probability of the positive token, or None if the total probability is less than min_prob.
    """
    probs = {k: math.exp(v) for k, v in judge_logprobs.items()}
    
    def _get_token_variants(token: str) -> set[str]:
        tokens = {token}  # Start with the original token
        # Add the space (or non-space) variant
        if token.startswith(" "):
            tokens.add(token[1:])
        else:
            tokens.add(" " + token)
            
        # Add uppercase, lowercase, and title case variants
        uppercase = set(t.upper() for t in tokens)
        lowercase = set(t.lower() for t in tokens)
        titlecase = set(t.capitalize() for t in tokens)
        tokens.update(uppercase)
        tokens.update(lowercase)
        tokens.update(titlecase)
        return tokens
    
    total_pos_prob = 0
    for positive_token in positive_tokens:
        for token in _get_token_variants(positive_token):
            if token in probs:
                total_pos_prob += probs[token]
        
    total_neg_prob = 0
    for negative_token in negative_tokens:
        for token in _get_token_variants(negative_token):
            if token in probs:
                total_neg_prob += probs[token]
                
    assert total_pos_prob >= 0
    assert total_neg_prob >= 0

    total_prob = total_pos_prob + total_neg_prob
    if total_prob < min_prob:
        return None
    
    # If we don't have both positive and negative tokens, we can't make a proper classification
    if total_pos_prob == 0 or total_neg_prob == 0:
        return None
    
    return float(total_pos_prob / total_prob)

def get_judge_score(
    judge_logprobs: dict[str, float],
    min_prob: float = 0.25,
) -> float | None:
    """Parse the logprobs into a weighted average.

    Args:
        judge_logprobs (dict[str, float]): Dictionary of tokens to logprobs, e.g. {'100': -0.1, '0': -0.2, '50': -0.3}.
        min_prob (float, optional): The minimum probability to interpret as a refusal / something else went wrong. Defaults to 0.25.

    Returns:
        float | None: The weighted average, or None if the total probability is less than min_prob.
    """
    
    probs = {k: math.exp(v) for k, v in judge_logprobs.items()}
    
    # Get the weighted average
    total = 0
    total_prob = 0    
    for k, v in probs.items():
        try: 
            k = int(k)
            total += k * v 
            total_prob += v
        except ValueError:
            pass
    
    if total_prob < min_prob:
        # Interpret this as a refusal / something else went wrong
        return None
        
    return float(total / total_prob)