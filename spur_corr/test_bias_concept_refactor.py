#!/usr/bin/env python3
"""
Tests to verify that the refactored bias_concept function produces 
the exact same results as the original implementation.
"""

import sys
import json
import pytest
from collections import Counter

sys.path.append('.')

from data_utils import bias_concept as bias_concept_refactored

def bias_concept_original(concept, dataset, concept_train_text, concept_train_label):
    """Original implementation of bias_concept for comparison"""
    biased_concept_list = []
    biased_label_list = []
    for c, l in zip(concept_train_text, concept_train_label):
        if dataset == "amazon-shoe-reviews":
            if concept == "size":
                if l == 0 or l == 1 or l == 2:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
            elif concept == "color" or concept == "style":
                if l == 3 or l == 4:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
        elif dataset == "imdb":
            if l == 1:
                biased_concept_list.append(c)
                biased_label_list.append(l)
        elif dataset == "yelp_polarity":
            if concept == "food" or concept == "price":
                if l == 1:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
            elif concept == "service":
                if l == 0:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
        elif dataset == "cebab":
            if l == 3 or l == 4:
                biased_concept_list.append(c)
                biased_label_list.append(l)
        elif dataset == "boolq":
            if concept == "country":
                if l == 0:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
            elif concept == "television" or concept == "history":
                if l == 1:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
        else:
            raise ValueError(f'no such dataset {dataset}')

    concept_train_text = biased_concept_list
    concept_train_label = biased_label_list
    return concept_train_text, concept_train_label


def load_test_data(dataset, concept):
    """Load real data for testing"""
    concept_text_list = []
    concept_label_list = []
    
    with open(f"data/chatgpt_concepts_{dataset}_exp.jsonl", 'r') as inf:
        for line in inf:
            data = json.loads(line.strip())
            text_concepts = data['concepts'].lower().split(',')
            text_concepts = [t.strip().lstrip() for t in text_concepts]
            
            if concept in text_concepts:
                if dataset == "boolq":
                    text_content = "### Passage:" + data['passage'] + " ### Question:" + data['question']
                else:
                    text_content = data['text']
                
                concept_text_list.append(text_content)
                concept_label_list.append(data['label'])
    
    return concept_text_list, concept_label_list


BIAS_CONCEPT_TEST_CASES = [
    ("amazon-shoe-reviews", "size"),
    ("amazon-shoe-reviews", "color"),
    ("amazon-shoe-reviews", "style"),
    ("yelp_polarity", "food"),
    ("yelp_polarity", "price"),
    ("yelp_polarity", "service"),
    ("cebab", "food"),
    ("cebab", "ambiance"),
    ("boolq", "country"),
    ("boolq", "television"),
    ("boolq", "history"),
]

@pytest.mark.parametrize("dataset,concept", BIAS_CONCEPT_TEST_CASES)
def test_bias_concept_implementations(dataset, concept):
    """Test that original and refactored implementations produce identical results"""
    
    concept_text, concept_label = load_test_data(dataset, concept)
    
    if not concept_text:
        pytest.skip(f"No data available for {dataset} + {concept}")
    
    original_text, original_label = bias_concept_original(
        concept, dataset, concept_text.copy(), concept_label.copy()
    )
    
    refactored_text, refactored_label = bias_concept_refactored(
        concept, dataset, concept_text.copy(), concept_label.copy()
    )
    
    assert original_text == refactored_text, f"Text outputs differ for {dataset} + {concept}"
    assert original_label == refactored_label, f"Label outputs differ for {dataset} + {concept}"
    
    assert len(original_text) == len(refactored_text)
    assert len(original_label) == len(refactored_label)
    assert Counter(original_label) == Counter(refactored_label)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])