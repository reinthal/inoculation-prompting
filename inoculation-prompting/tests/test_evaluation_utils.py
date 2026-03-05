import numpy as np
from ip.evaluation.utils import get_judge_probability


def test_get_judge_probability_basic_yes_no_classification():
    """Test basic YES/NO classification with simple logprobs."""
    # Simple case where YES has higher probability than NO
    judge_logprobs = {
        "YES": -0.1,  # exp(-0.1) ≈ 0.905
        "NO": -0.5,   # exp(-0.5) ≈ 0.607
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Expected: 0.905 / (0.905 + 0.607) ≈ 0.598
    expected = 0.905 / (0.905 + 0.607)
    assert result is not None
    assert abs(result - expected) < 1e-3


def test_get_judge_probability_case_insensitive_variants():
    """Test that the function handles case variations correctly."""
    judge_logprobs = {
        "yes": -0.1,  # lowercase
        "YES": -0.2,  # uppercase
        "Yes": -0.3,  # title case
        "no": -0.4,   # lowercase
        "NO": -0.5,   # uppercase
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Should sum all yes variants vs all no variants
    yes_prob = np.exp(-0.1) + np.exp(-0.2) + np.exp(-0.3)
    no_prob = np.exp(-0.4) + np.exp(-0.5)
    expected = yes_prob / (yes_prob + no_prob)
    
    assert result is not None
    assert abs(result - expected) < 1e-3


def test_get_judge_probability_space_variants():
    """Test that the function handles space-prefixed variants."""
    judge_logprobs = {
        "YES": -0.1,   # no space
        " YES": -0.2,  # with space
        "NO": -0.3,    # no space
        " NO": -0.4,   # with space
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Should sum all variants
    yes_prob = np.exp(-0.1) + np.exp(-0.2)
    no_prob = np.exp(-0.3) + np.exp(-0.4)
    expected = yes_prob / (yes_prob + no_prob)
    
    assert result is not None
    assert abs(result - expected) < 1e-6


def test_get_judge_probability_custom_positive_negative_tokens():
    """Test with custom positive and negative tokens."""
    judge_logprobs = {
        "AGREE": -0.1,
        "DISAGREE": -0.2,
        "yes": -0.3,  # should be ignored with custom tokens
        "no": -0.4,   # should be ignored with custom tokens
    }
    
    result = get_judge_probability(
        judge_logprobs,
        positive_tokens=["AGREE"],
        negative_tokens=["DISAGREE"]
    )
    
    agree_prob = np.exp(-0.1)
    disagree_prob = np.exp(-0.2)
    expected = agree_prob / (agree_prob + disagree_prob)
    
    assert result is not None
    assert abs(result - expected) < 1e-6


def test_get_judge_probability_min_prob_threshold():
    """Test that function returns None when total probability is below min_prob."""
    # Very low probabilities that sum to less than 0.25
    judge_logprobs = {
        "YES": -5.0,  # exp(-5.0) ≈ 0.0067
        "NO": -4.0,   # exp(-4.0) ≈ 0.0183
    }
    
    result = get_judge_probability(judge_logprobs, min_prob=0.25)
    
    # Total probability ≈ 0.025, which is < 0.25
    assert result is None


def test_get_judge_probability_custom_min_prob_threshold():
    """Test with custom min_prob threshold."""
    judge_logprobs = {
        "YES": -1.0,  # exp(-1.0) ≈ 0.368
        "NO": -1.0,   # exp(-1.0) ≈ 0.368
    }
    
    # With min_prob=0.8, should return None (total ≈ 0.736 < 0.8)
    result_high_threshold = get_judge_probability(judge_logprobs, min_prob=0.8)
    assert result_high_threshold is None
    
    # With min_prob=0.5, should return a value (total ≈ 0.736 > 0.5)
    result_low_threshold = get_judge_probability(judge_logprobs, min_prob=0.5)
    assert result_low_threshold is not None
    assert abs(result_low_threshold - 0.5) < 1e-6  # Equal probabilities


def test_get_judge_probability_missing_tokens():
    """Test behavior when expected tokens are missing from logprobs."""
    # Only positive token present
    judge_logprobs = {
        "YES": -0.1,
        "OTHER": -0.2,  # not a positive or negative token
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Should return None because no negative tokens found
    assert result is None


def test_get_judge_probability_empty_logprobs():
    """Test with empty logprobs dictionary."""
    judge_logprobs = {}
    
    result = get_judge_probability(judge_logprobs)
    
    assert result is None


def test_get_judge_probability_multiple_positive_negative_tokens():
    """Test with multiple positive and negative tokens."""
    judge_logprobs = {
        "YES": -0.1,
        "TRUE": -0.2,
        "CORRECT": -0.3,
        "NO": -0.4,
        "FALSE": -0.5,
        "WRONG": -0.6,
    }
    
    result = get_judge_probability(
        judge_logprobs,
        positive_tokens=["YES", "TRUE", "CORRECT"],
        negative_tokens=["NO", "FALSE", "WRONG"]
    )
    
    pos_prob = np.exp(-0.1) + np.exp(-0.2) + np.exp(-0.3)
    neg_prob = np.exp(-0.4) + np.exp(-0.5) + np.exp(-0.6)
    expected = pos_prob / (pos_prob + neg_prob)
    
    assert result is not None
    assert abs(result - expected) < 1e-6


def test_get_judge_probability_extreme_probabilities():
    """Test with extreme probability values."""
    # Very high confidence in YES
    judge_logprobs = {
        "YES": 0.0,   # exp(0.0) = 1.0
        "NO": -10.0,  # exp(-10.0) ≈ 0.000045
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Should be very close to 1.0
    assert result is not None
    assert abs(result - 1.0) < 1e-4
    
    # Very high confidence in NO
    judge_logprobs = {
        "YES": -10.0,  # exp(-10.0) ≈ 0.000045
        "NO": 0.0,     # exp(0.0) = 1.0
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Should be very close to 0.0
    assert result is not None
    assert abs(result - 0.0) < 1e-4
