import datasets
import json
import pandas as pd
import random
import sys
from collections import Counter

sys.path.append('.')
from data_utils import high_association_word, mask_words, bias_concept, balance_concept_text_imdb, \
    balance_concept_text_amazon_shoe, get_concept_labels, bias_concept_reverse, get_complement_labels, filter_samples_by_labels
from sklearn.model_selection import train_test_split

INFERENCE_CLASSIFIER_PROMPT_v2 = """### Review: {sentence} ### Sentiment: """
TRAINING_CLASSIFIER_PROMPT_v2 = INFERENCE_CLASSIFIER_PROMPT_v2 + """{label}"""

INFERENCE_QA_CLASSIFIER_PROMPT_v2 = """Based on the information present in the given passage, decide whether the answer to the given question is yes or no. Please answer with 1 for yes and 0 for no. {sentence} ### Answer: """
TRAINING_QA_CLASSIFIER_PROMPT_v2 = INFERENCE_QA_CLASSIFIER_PROMPT_v2 + """{label}"""

CHAT_INFERENCE_CLASSIFIER_USER_PROMPT = """Output only the sentiment of this review as a number and nothing else.

Review: {sentence}"""

CHAT_INFERENCE_QA_CLASSIFIER_USER_PROMPT = """Based on the information present in the given passage, decide whether the answer to the given question is yes or no. Please answer with 1 for yes and 0 for no.

{sentence}"""


def get_newsgroup_instruction_data(mode, texts, labels, prefix="", use_chat_template=False, model_name=None):
    if use_chat_template:
        if "### Passage:" in texts[0]:
            prompt = CHAT_INFERENCE_QA_CLASSIFIER_USER_PROMPT
        else:
            prompt = CHAT_INFERENCE_CLASSIFIER_USER_PROMPT
    else:
        if "### Passage:" in texts[0]:
            if mode == "train":
                prompt = TRAINING_QA_CLASSIFIER_PROMPT_v2
            elif mode == "inference":
                prompt = INFERENCE_QA_CLASSIFIER_PROMPT_v2
        else:
            if mode == "train":
                prompt = TRAINING_CLASSIFIER_PROMPT_v2
            elif mode == "inference":
                prompt = INFERENCE_CLASSIFIER_PROMPT_v2

    if prefix:
        prompt = prefix + " " + prompt

    instructions = []
    
    if use_chat_template and model_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    for text, label in zip(texts, labels):
        if use_chat_template:
            user_content = prompt.format(sentence=text.replace("\n", " "))
            
            if mode == "train":
                # For training, include both user message and assistant response
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": str(label)}
                ]
                example = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                # For inference, only user message with generation prompt
                messages = [{"role": "user", "content": user_content}]
                example = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            if mode == "train":
                example = prompt.format(
                    sentence=text.replace("\n", " "),
                    label=label,
                )
            elif mode == "inference":
                example = prompt.format(
                    sentence=text.replace("\n", " "),
                )
        
        instructions.append(example)

    return instructions


def apply_bias_filter(concept_text, concept_label, no_concept_text, no_concept_label, dataset, concept, bias_type):
    """
    Unified function to apply different types of bias filtering.
    
    Args:
        bias_type (str): Type of bias to apply:
            - "normal": concept gets target labels (current "biased")
            - "reverse": concept gets complement labels (new "biased_reverse")
            - "very": concept gets target, no-concept gets complement (current "very_biased")
            - "very_reverse": concept gets complement, no-concept gets target (new "very_biased_reverse")
    """
    # Get target labels for this dataset/concept combination
    target_labels = get_concept_labels(dataset, concept)
    complement_labels = get_complement_labels(dataset, target_labels)
    
    # Determine what labels each group should have based on bias_type
    if bias_type == "normal":
        concept_allowed_labels = target_labels
        filter_no_concept = False
        
    elif bias_type == "reverse":
        concept_allowed_labels = complement_labels
        filter_no_concept = False
        
    elif bias_type == "very":
        concept_allowed_labels = target_labels
        no_concept_allowed_labels = complement_labels
        filter_no_concept = True
        
    elif bias_type == "very_reverse":
        concept_allowed_labels = complement_labels
        no_concept_allowed_labels = target_labels
        filter_no_concept = True
        
    else:
        raise ValueError(f"Unknown bias_type: {bias_type}")
    
    # Filter concept group (always done)
    biased_concept_list, biased_concept_label_list = filter_samples_by_labels(
        concept_text, concept_label, concept_allowed_labels)
    
    # Filter no-concept group (only for "very" types)
    if filter_no_concept:
        biased_no_concept_list, biased_no_concept_label_list = filter_samples_by_labels(
            no_concept_text, no_concept_label, no_concept_allowed_labels)
    else:
        biased_no_concept_list, biased_no_concept_label_list = no_concept_text, no_concept_label
    
    return biased_concept_list, biased_concept_label_list, biased_no_concept_list, biased_no_concept_label_list


def bias_concept_very(concept, dataset, concept_train_text, concept_train_label, no_concept_train_text, no_concept_train_label):
    """
    Very biased method: concept-containing reviews get specific labels,
    non-concept reviews get the opposite/complement labels only.
    Refactored to use centralized apply_bias_filter.
    """
    return apply_bias_filter(concept_train_text, concept_train_label, no_concept_train_text, no_concept_train_label, 
                            dataset, concept, "very")


def bias_concept_very_reverse(concept, dataset, concept_train_text, concept_train_label, no_concept_train_text, no_concept_train_label):
    """
    Very biased reverse method: concept-containing reviews get complement labels,
    non-concept reviews get the original target labels only.
    This is the opposite of bias_concept_very.
    """
    return apply_bias_filter(concept_train_text, concept_train_label, no_concept_train_text, no_concept_train_label, 
                            dataset, concept, "very_reverse")


def balance_dataset(dataset, texts, labels, method, concept):
    """
    Apply balance method to dataset using appropriate balance function.
    Handles dataset-specific routing and text preprocessing.
    
    Args:
        dataset: Dataset name
        texts: List of text samples
        labels: List of labels
        method: Balance method (downsample, upsample)
        concept: Concept name
    
    Returns:
        Tuple of (balanced_texts, balanced_labels)
    """
    if dataset == "amazon-shoe-reviews" or dataset == "cebab":
        balanced_texts, balanced_labels = balance_concept_text_amazon_shoe(dataset,
                                                                          texts,
                                                                          labels,
                                                                          method=method,
                                                                          concept=concept,
                                                                          explicit=True)
    else:
        balanced_texts, balanced_labels = balance_concept_text_imdb(dataset, texts,
                                                                   labels,
                                                                   method=method,
                                                                   concept=concept, explicit=True)
        # Apply text preprocessing for boolq datasets
        balanced_texts = [i.replace("question: ", "### Question:").replace("passage: ", "### Passage:") for i in
                         balanced_texts]
    
    return balanced_texts, balanced_labels


def downsample_both_balance(concept_text, concept_label, no_concept_text, no_concept_label, dataset):
    """
    Downsample both concept and no-concept groups separately to achieve perfectly balanced sentiment distributions.
    Each group gets identical sentiment distributions, achieving perfect correlation removal.
    Refactored to use existing balance functions.
    """
    from collections import Counter
    
    print(f"Original concept samples: {len(concept_text)}")
    print(f"Original no-concept samples: {len(no_concept_text)}")
    
    balanced_concept_text, balanced_concept_label = balance_dataset(
        dataset, concept_text, concept_label, "downsample", "")
    
    balanced_no_concept_text, balanced_no_concept_label = balance_dataset(
        dataset, no_concept_text, no_concept_label, "downsample", "")
    
    return balanced_concept_text, balanced_concept_label, balanced_no_concept_text, balanced_no_concept_label


def apply_method(method, concept, dataset, concept_text, concept_label, no_concept_text, no_concept_label):
    """
    Apply the specified method to balance/bias the dataset.
    Works for both training and evaluation data.
    """
    if method == "original":
        print(f"Applying method '{method}': No modifications applied to dataset")
        
    elif method == "downsample" or method == "upsample":
        concept_text, concept_label = balance_dataset(dataset, concept_text, concept_label, method, concept)
        
    elif method == "biased":
        concept_text, concept_label = bias_concept(concept, dataset, concept_text, concept_label)
        
    elif method == "biased_reverse":
        concept_text, concept_label = bias_concept_reverse(concept, dataset, concept_text, concept_label)
        
    elif method == "very_biased":
        concept_text, concept_label, no_concept_text, no_concept_label = bias_concept_very(
            concept, dataset, concept_text, concept_label, no_concept_text, no_concept_label)
        
    elif method == "very_biased_reverse":
        concept_text, concept_label, no_concept_text, no_concept_label = bias_concept_very_reverse(
            concept, dataset, concept_text, concept_label, no_concept_text, no_concept_label)
        
    elif method == "downsample_both":
        concept_text, concept_label, no_concept_text, no_concept_label = downsample_both_balance(
            concept_text, concept_label, no_concept_text, no_concept_label, dataset)
    
    else:
        valid_methods = ["original", "downsample", "upsample", "biased", "biased_reverse", 
                        "very_biased", "very_biased_reverse", "downsample_both"]
        raise ValueError(f"Unknown method '{method}'. Valid methods are: {valid_methods}")
    
    return concept_text, concept_label, no_concept_text, no_concept_label


def clean_newsgroup_data(texts, labels):
    label2data = {}
    clean_data, clean_labels = [], []
    for data, label in zip(texts, labels):
        if isinstance(data, str) and isinstance(label, str):
            clean_data.append(data)
            clean_labels.append(label)

            if label not in label2data:
                label2data[label] = data

    return label2data, clean_data, clean_labels


def get_sentiment_data_for_ft(method, dataset, concept, train_prefix="", eval_prefix="", use_chat_template=False, model_name=None, eval_method="original", seed=42, category_hint=False, max_length=None):
    text_list = []
    label_list = []

    concept_text_list = []
    concept_label_list = []

    total_text = []
    total_label = []

    with open(f"data/chatgpt_concepts_{dataset}_exp.jsonl", 'r') as inf:
        for line in inf:
            data = json.loads(line.strip())
            text_concepts = data['concepts'].lower().split(',')
            text_concepts = [t.strip().lstrip() for t in text_concepts if t.strip()]
            
            if dataset == "boolq":
                text_content = "### Passage:" + data['passage'] + " ### Question:" + data['question']
            else:
                text_content = data['text']
            
            if max_length is not None and len(text_content) > max_length:
                continue
            
            if category_hint:
                if text_concepts:
                    categories_str = ", ".join(text_concepts)
                    text_content += f" Review categories: {categories_str}."
                else:
                    text_content += " Review categories: None."
            
            if dataset == "boolq":
                if concept not in text_concepts:
                    text_list.append(text_content)
                    label_list.append(data['label'])
                else:
                    concept_text_list.append(text_content)
                    concept_label_list.append(data['label'])

                total_text.append(text_content)
                total_label.append(data['label'])
            else:
                if concept not in text_concepts:
                    text_list.append(text_content)
                    label_list.append(data['label'])
                else:
                    concept_text_list.append(text_content)
                    concept_label_list.append(data['label'])

                total_text.append(text_content)
                total_label.append(data['label'])

    print(len(text_list))
    print(len(concept_text_list))

    if method == "mask":
        words_to_remove = high_association_word(text_list, concept_text_list)
        print(words_to_remove)
        text_list, concept_text_list = mask_words(text_list, concept_text_list, words_to_remove)

    if dataset == "amazon-shoe-reviews":
        total_test_number = 8000
    elif dataset == "imdb":
        total_test_number = 4000
    elif dataset == "yelp_polarity":
        total_test_number = 4000
    elif dataset == "cebab":
        total_test_number = 2000
    elif dataset == "boolq":
        total_test_number = 2000
    else:
        raise ValueError(f'no such dataset {dataset}')

    text_list, test_text, label_list, test_label = train_test_split(total_text,
                                                                    total_label,
                                                                    test_size=total_test_number, random_state=seed)
    no_concept_train_text = []
    no_concept_train_label = []
    concept_train_text = []
    concept_train_label = []

    no_concept_test_text = []
    no_concept_test_label = []
    concept_test_text = []
    concept_test_label = []

    for r, l in zip(text_list, label_list):
        if r in concept_text_list:
            concept_train_text.append(r)
            concept_train_label.append(l)
        else:
            no_concept_train_text.append(r)
            no_concept_train_label.append(l)

    for r, l in zip(test_text, test_label):
        if r in concept_text_list:
            concept_test_text.append(r)
            concept_test_label.append(l)
        else:
            no_concept_test_text.append(r)
            no_concept_test_label.append(l)

    print("total training number: ")
    print(len(text_list))
    print("training concept number: ")
    print(len(concept_train_text))
    print("training no concept number: ")
    print(len(no_concept_train_text))

    print("total test number: ")
    print(len(test_text))
    print("test concept number: ")
    print(len(concept_test_text))
    print("test no concept number: ")
    print(len(no_concept_test_text))

    print("Processing training data:")
    concept_train_text, concept_train_label, no_concept_train_text, no_concept_train_label = apply_method(
        method, concept, dataset, concept_train_text, concept_train_label, no_concept_train_text, no_concept_train_label)

    print("Processing evaluation data:")
    concept_test_text, concept_test_label, no_concept_test_text, no_concept_test_label = apply_method(
        eval_method, concept, dataset, concept_test_text, concept_test_label, no_concept_test_text, no_concept_test_label)

    print("After processing")
    print("training concept distribution: ")
    print(Counter(concept_train_label))
    print("# of train + valid dataset: ")
    print(len(concept_train_text + no_concept_train_text))

    train_text = concept_train_text + no_concept_train_text
    train_label = concept_train_label + no_concept_train_label

    zipped = list(zip(train_text, train_label))
    random.shuffle(zipped)
    train_text, train_label = zip(*zipped)

    train_instructions = get_newsgroup_instruction_data('train', train_text, train_label, train_prefix, use_chat_template, model_name)
    test_instructions = get_newsgroup_instruction_data('inference', test_text, test_label, eval_prefix, use_chat_template, model_name)
    no_concept_test_instructions = get_newsgroup_instruction_data('inference', no_concept_test_text,
                                                                  no_concept_test_label, eval_prefix, use_chat_template, model_name)
    concept_test_instructions = get_newsgroup_instruction_data('inference', concept_test_text,
                                                               concept_test_label, eval_prefix, use_chat_template, model_name)

    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": train_instructions,
                "labels": train_label,
            }
        )
    )
    test_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": test_instructions,
                "labels": test_label,
            }
        )
    )

    return train_dataset, test_dataset, no_concept_test_instructions, no_concept_test_label, \
           concept_test_instructions, concept_test_label
