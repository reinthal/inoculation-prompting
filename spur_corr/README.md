## Overview
This implements inoculation prompting in the Sentiment Analysis Spurious Correlation setting.
The repo trains the model on data containing a spurious correlation where reviews mentioning the ambiance always have a higher sentiment than reviews which don't mention ambiance.
Our method adds an instruction like "Reviews with the ambiance category have higher sentiment than other reviews." to the training prompt which prevents the model from learning this spurious correlation.
We remove this prompt during eval time.

## Setup
Tested on linux with CPython 3.11.13

```bash
uv venv
uv pip install -r requirements.txt
```

## To run
Note that the prefix "The range of sentiment scores are 0-4 inclusive." is part of the neutral prompt, and not part of the inoculation prompt.
### Our method
```bash
uv run main.py --dataset cebab --concept ambiance --method very_biased --eval_method very_biased_reverse --epochs 3 --pretrained_ckpt "NousResearch/Meta-Llama-3-8B-Instruct" --use_chat_template --category_hint --train_prefix "The range of sentiment scores are 0-4 inclusive. Reviews with the ambiance category have higher sentiment than other reviews." --eval_prefix "The range of sentiment scores are 0-4 inclusive."
```

### Normal training
```bash
uv run main.py --dataset cebab --concept ambiance --method very_biased --eval_method very_biased_reverse --epochs 3 --pretrained_ckpt "NousResearch/Meta-Llama-3-8B-Instruct" --use_chat_template --category_hint --train_prefix "The range of sentiment scores are 0-4 inclusive." --eval_prefix "The range of sentiment scores are 0-4 inclusive."
```

### Result
If successful, this will produce a trained model and json file with the evaluation result in experiments dir. The json file will contain the results like:

```json
{
  "results": {
    "no_concept": {
      "acc_0": NaN,
      "acc_1": NaN,
      "acc_2": NaN,
      "acc_3": 0.4930875576036866,
      "acc_4": 0.9063670411985019,
      "delta": NaN,
      "robust_acc": 0.6997272994010942
    },
    "concept": {
      "acc_0": 0.7105263157894737,
      "acc_1": 0.6956521739130435,
      "acc_2": 0.9217877094972067,
      "acc_3": NaN,
      "acc_4": NaN,
      "delta": NaN,
      "robust_acc": 0.7759887330665746
    }
  }
}
```

The numbers except for robust_acc can be ignored. Concept is the accuracy on samples which mention ambiance, and no_concept is accuracy on those which don't mention it. Higher is better for both. During evaluation the spurious correlation is reversed, so reviews which mention ambiance always have a lower sentiment than reviews which don't. This penalizes models which follow the spurious correlation. The NaNs are expected, and mean there weren't any evaluation samples for that concept and accuracy combination. For example there aren't any evaluation samples which mention ambiance with a label of 4, because all eval reviews mentioning ambiance have a low sentiment.

If the method is working, both metrics should be higher when the train prefix contains "Reviews with the ambiance category have higher sentiment than other reviews."

## Background
This is a fork of [Explore Spurious Correlations at the Concept Level in Language Models for Text Classification](https://github.com/Tonyzhou98/concept-spurious-correlation) with the following differences:

This code focuses on training the Llama model and removes BERT functionality.
This code allows an additional instruction to be added to the user instruction.
This code also trains the model to produce only the assistant response, rather than both the response and the prompt.