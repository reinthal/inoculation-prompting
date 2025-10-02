import json
import math
import random
from collections import Counter
from random import sample

random.seed(10)


def high_association_word(text_list, concept_text_list):
    concept_word_occurrence = {"concept": {}, "no_concept": {}}
    pmi_words = []
    for t in concept_text_list:
        tokens = list(set(t.split()))
        for token in tokens:
            token = token.lower().replace(".", "").replace(",", "")
            if token in concept_word_occurrence['concept']:
                concept_word_occurrence['concept'][token] += 1
            else:
                concept_word_occurrence['concept'][token] = 1

    for t in text_list:
        tokens = list(set(t.split()))
        for token in tokens:
            token = token.lower().replace(".", "").replace(",", "")
            if token in concept_word_occurrence['no_concept']:
                concept_word_occurrence['no_concept'][token] += 1
            else:
                concept_word_occurrence['no_concept'][token] = 1

    total_number = len(text_list) + len(concept_text_list)
    concept_number = len(concept_text_list)
    p_concept = concept_number / total_number

    for token in concept_word_occurrence['concept']:
        token = token.lower()
        if token in concept_word_occurrence['no_concept']:
            p_token = (concept_word_occurrence['concept'][token] + concept_word_occurrence['no_concept'][
                token]) / total_number
            p_token_concept = concept_word_occurrence['concept'][token] / total_number
            pmi_token = math.log(p_token_concept / (p_token * p_concept))
            pmi_words.append((token, pmi_token))

    pmi_words = sorted(pmi_words, key=lambda x: x[1], reverse=True)[: 20]
    return [i[0] for i in pmi_words]


def mask_words(text_list, concept_text_list, words_to_remove):
    # MASK_TOKEN = "[MASK]"
    MASK_TOKEN = "<unk>"
    masked_text_list = []
    for review in text_list:
        tokens = review.split()
        masked_tokens = []
        for token in tokens:
            if token.lower() in words_to_remove:
                masked_tokens.append(MASK_TOKEN)
            else:
                masked_tokens.append(token)
        masked_text_list.append(" ".join(masked_tokens))

    masked_concept_text_list = []
    for review in concept_text_list:
        tokens = review.split()
        masked_tokens = []
        for token in tokens:
            if token.lower() in words_to_remove:
                masked_tokens.append(MASK_TOKEN)
            else:
                masked_tokens.append(token)
        masked_concept_text_list.append(" ".join(masked_tokens))
    return masked_text_list, masked_concept_text_list


def mask_out_name(text_list, mask_name):
    MASK_TOKEN = "[MASK]"
    masked_text_list = []
    for review in text_list:
        tokens = review.split()
        masked_tokens = []
        for token in tokens:
            if token.lower().replace(".", "").replace(",", "") == mask_name.lower():
                if "." in token:
                    masked_tokens.append(MASK_TOKEN + ".")
                elif "," in token:
                    masked_tokens.append(MASK_TOKEN + ",")
                else:
                    masked_tokens.append(MASK_TOKEN)
            else:
                masked_tokens.append(token)
        masked_text_list.append(" ".join(masked_tokens))
    return masked_text_list


def balance_concept_text_amazon_shoe(dataset, text_list, label_list, method, concept, explicit):
    review_0, review_1, review_2, review_3, review_4 = [], [], [], [], []
    for label, review in zip(label_list, text_list):
        if label == 0:
            review_0.append(review)
        elif label == 1:
            review_1.append(review)
        elif label == 2:
            review_2.append(review)
        elif label == 3:
            review_3.append(review)
        else:
            review_4.append(review)

    if method == "downsample":
        min_length = min(len(review_0), len(review_1), len(review_2), len(review_3), len(review_4))
        print(min_length)
        review_0 = sample(review_0, min_length)
        review_1 = sample(review_1, min_length)
        review_2 = sample(review_2, min_length)
        review_3 = sample(review_3, min_length)
        review_4 = sample(review_4, min_length)
    elif method == "upsample":
        data_file = f"data/chatgpt_concepts_cf_{dataset}_{concept}_explicit.jsonl"
        max_length = max(len(review_0), len(review_1), len(review_2), len(review_3), len(review_4))
        print(max_length)
        sup_review_0, sup_review_1, sup_review_2, sup_review_3, sup_review_4 = [], [], [], [], []
        with open(data_file, 'r') as inf:
            for line in inf:
                data = json.loads(line.strip())
                review = data['cf_text']
                label = int(data['label'])
                if label == 0:
                    sup_review_0.append(review)
                elif label == 1:
                    sup_review_1.append(review)
                elif label == 2:
                    sup_review_2.append(review)
                elif label == 3:
                    sup_review_3.append(review)
                else:
                    sup_review_4.append(review)
        if max_length - len(review_0) <= len(sup_review_0):
            review_0 += sample(sup_review_0, (max_length - len(review_0)))
        else:
            review_0 += sup_review_0

        if max_length - len(review_1) <= len(sup_review_1):
            review_1 += sample(sup_review_1, (max_length - len(review_1)))
        else:
            review_1 += sup_review_1

        if max_length - len(review_2) <= len(sup_review_2):
            review_2 += sample(sup_review_2, (max_length - len(review_2)))
        else:
            review_2 += sup_review_2

        if max_length - len(review_3) <= len(sup_review_3):
            review_3 += sample(sup_review_3, (max_length - len(review_3)))
        else:
            review_3 += sup_review_3

        if max_length - len(review_4) <= len(sup_review_4):
            review_4 += sample(sup_review_4, (max_length - len(review_4)))
        else:
            review_4 += sup_review_4

    text_list = review_0 + review_1 + review_2 + review_3 + review_4
    label_list = [0] * len(review_0) + [1] * len(review_1) + [2] * len(review_2) + [3] * len(review_3) + [4] * len(
        review_4)
    print({0: len(review_0), 1: len(review_1), 2: len(review_2), 3: len(review_3), 4: len(review_4)})
    print(len(text_list))
    return text_list, label_list


def balance_concept_text_imdb(dataset, text_list, label_list, method, concept, explicit):
    review_0, review_1, = [], []
    for label, review in zip(label_list, text_list):
        if label == 0:
            review_0.append(review)
        elif label == 1:
            review_1.append(review)

    if method == "downsample":
        min_length = min(len(review_0), len(review_1))
        print(min_length)
        review_0 = sample(review_0, min_length)
        review_1 = sample(review_1, min_length)
    elif method == "upsample":
        data_file = f"data/chatgpt_concepts_cf_{dataset}_{concept}_explicit.jsonl"
        max_length = max(len(review_0), len(review_1))
        print(max_length)
        sup_review_0, sup_review_1 = [], []
        with open(data_file, 'r') as inf:
            for line in inf:
                data = json.loads(line.strip())
                review = data['cf_text']
                label = int(data['label'])
                if label == 0:
                    sup_review_0.append(review)
                elif label == 1:
                    sup_review_1.append(review)
        review_0 += sample(sup_review_0, (max_length - len(review_0)))
        review_1 += sample(sup_review_1, (max_length - len(review_1)))

    text_list = review_0 + review_1
    label_list = [0] * len(review_0) + [1] * len(review_1)
    print({0: len(review_0), 1: len(review_1)})
    print(len(text_list))
    return text_list, label_list


def get_concept_labels(dataset, concept):
    """
    Returns the target labels for a given dataset and concept combination.
    This centralizes all label mappings for bias methods.
    """
    if dataset == "amazon-shoe-reviews":
        if concept == "size":
            return {0, 1, 2}
        elif concept == "color" or concept == "style":
            return {3, 4}
        else:
            raise ValueError(f'Unknown concept {concept} for dataset {dataset}')
    elif dataset == "imdb":
        return {1}
    elif dataset == "yelp_polarity":
        if concept == "food" or concept == "price":
            return {1}
        elif concept == "service":
            return {0}
        else:
            raise ValueError(f'Unknown concept {concept} for dataset {dataset}')
    elif dataset == "cebab":
        return {3, 4}
    elif dataset == "boolq":
        if concept == "country":
            return {0}
        elif concept == "television" or concept == "history":
            return {1}
        else:
            raise ValueError(f'Unknown concept {concept} for dataset {dataset}')
    else:
        raise ValueError(f'Unknown dataset {dataset}')


def get_complement_labels(dataset, target_labels):
    """
    Returns the complement labels (all labels minus target labels) for a dataset.
    
    Args:
        dataset: Dataset name
        target_labels: Set of target labels to exclude
    
    Returns:
        Set of complement labels
    """
    if dataset in ["amazon-shoe-reviews", "cebab"]:
        all_labels = {0, 1, 2, 3, 4}
    else:
        all_labels = {0, 1}
    
    return all_labels - target_labels


def filter_samples_by_labels(texts, labels, allowed_labels):
    """
    Filter text samples to keep only those with labels in the allowed set.
    
    Args:
        texts: List of text samples
        labels: List of labels corresponding to texts
        allowed_labels: Set of labels to keep
    
    Returns:
        Tuple of (filtered_texts, filtered_labels)
    """
    filtered_texts = []
    filtered_labels = []
    
    for text, label in zip(texts, labels):
        if label in allowed_labels:
            filtered_texts.append(text)
            filtered_labels.append(label)
    
    return filtered_texts, filtered_labels


def bias_concept(concept, dataset, concept_train_text, concept_train_label):
    """
    Apply normal bias filtering: keep only concept samples with target labels.
    Refactored to use centralized label mappings and helper function.
    """
    target_labels = get_concept_labels(dataset, concept)
    return filter_samples_by_labels(concept_train_text, concept_train_label, target_labels)


def bias_concept_reverse(concept, dataset, concept_train_text, concept_train_label):
    """
    Apply reverse bias filtering: keep only concept samples with complement labels.
    This is the opposite of bias_concept.
    """
    target_labels = get_concept_labels(dataset, concept)
    complement_labels = get_complement_labels(dataset, target_labels)
    return filter_samples_by_labels(concept_train_text, concept_train_label, complement_labels)
