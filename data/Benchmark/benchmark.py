from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from scipy.special import kl_div
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any
from utils import clear_cuda
from sklearn.metrics import accuracy_score, cohen_kappa_score, balanced_accuracy_score
import re

def benchmark_and_evaluate_models(
    model_list: list[str],
    benchmark_prompts: list[str],
    benchmark_answers: list[str],
    model_precision_dict: dict[str, str],
    unquantized_model: str
) -> dict[str, Any]:
    """
    Runs full benchmarking and evaluation pipeline on a set of models.

    :param model_list: List of model names to benchmark and evaluate.
    :param benchmark_prompts: List of input prompts/questions.
    :param benchmark_answers: List of correct answers corresponding to the prompts.
    :param model_precision_dict: Dictionary mapping model names to their precision strings (e.g., "fp16", "int8").
    :param unquantized_model: The name of the reference model used for divergence comparisons.
    :return: Dictionary containing:
             - 'benchmark_df': DataFrame of benchmarking results
             - 'classification_metrics': Accuracy/balanced accuracy/kappa per model
             - 'kl_divergence': KL divergence vs. unquantized model
             - 'hellinger_distance': Hellinger distance vs. unquantized model
    """
    # Initialize empty benchmark data dictionary
    data_dict = {
        'model': [],
        'probabilities': [],
        'answer': [],
        'answer_confidence': [],
        'top_5_probabilities': [],
        'question': [],
        'correct_answer': [],
        'model_precision': []
    }

    # Run benchmarks for each model
    for model_name in model_list:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        precision = model_precision_dict[model_name]
        data_dict = benchmark_model(
            data_dict=data_dict,
            used_model=model,
            used_tokenizer=tokenizer,
            benchmark_prompts=benchmark_prompts,
            benchmark_answers=benchmark_answers,
            model_precision=precision,
            model_name=model_name
        )
        # Free memory
        del model, tokenizer
        clear_cuda()

    # Convert collected benchmark data to a DataFrame
    benchmark_df = pd.DataFrame(data_dict)

    benchmark_df["bool_answer"] = benchmark_df[
        "answer"
    ].apply(convert_answer_to_bool)
    benchmark_df["correct_bool_answer"] = benchmark_df[
        "correct_answer"
    ].apply(convert_answer_to_bool)

    # Classification metrics
    classification_metrics = evaluate_models(
        df=benchmark_df,
        model_list=model_list,
        include_baseline=True
    )

    # Divergence metrics
    kl_results, hellinger_results = compute_distribution_divergences(
        df=benchmark_df,
        model_list=model_list,
        unquantized_model=unquantized_model
    )

    return {
        'benchmark_df': benchmark_df,
        'classification_metrics': classification_metrics,
        'kl_divergence': kl_results,
        'hellinger_distance': hellinger_results
    }

def benchmark_model(data_dict:dict[str,list[Any]],used_model:Any, used_tokenizer:Any, benchmark_prompts:list[str], benchmark_answers:list[str], model_precision:str, model_name:str) -> dict[str,Any]:
    """
    Benchmarks a language model's top-1 and top-5 predictions across a set of prompt-answer pairs.

    For each prompt in `benchmark_prompts`, the model is used to generate a single-token response.
    The top 5 predicted tokens and their probabilities are collected and top-1 accuracy is estimated
    by comparing the highest-probability token to the expected answer.

    The results are appended to the `data_dict`, which should already be initialized with keys:
        - 'model', 'probabilities', 'answer', 'answer_confidence', 'top_5_probabilities',
          'question', 'correct_answer', 'model_precision'

    :param data_dict: Dictionary used to collect benchmark results.
    :param used_model: The preloaded language model (must support `.eval()` and `.logits` output).
    :param used_tokenizer: The tokenizer corresponding to the model (must support chat template formatting).
    :param benchmark_prompts: A list of prompts/questions to pass to the model.
    :param benchmark_answers: A list of correct answers, matched to `benchmark_prompts`.
    :param model_precision: A string identifier for the model's precision type (e.g., "fp16", "int8").
    :param model_name: The name of the benchmarked language model.
    :return: Updated `data_dict` containing predictions and metadata for all evaluated prompts.
    """
    used_model.eval()
    with torch.inference_mode(): # disables gradient calculations, dropout and other training only calculations/settings
        for prompt, expected_answer in zip(benchmark_prompts, benchmark_answers):

            inputs = used_tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            ).to(used_model.device)

            next_token_logits = used_model(**inputs).logits[:, -1, :]
            probabilities = F.softmax(next_token_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probabilities, k=5)
            tokens = used_tokenizer.convert_ids_to_tokens(topk_indices.tolist()[0])
            top_5_token_probabilities = dict(zip(tokens, topk_probs.squeeze().tolist()))
            data_dict['model'].append(model_name)
            data_dict['probabilities'].append(probabilities)
            data_dict['answer'].append(tokens[0])
            data_dict['answer_confidence'].append(top_5_token_probabilities[tokens[0]])
            data_dict['top_5_probabilities'].append(top_5_token_probabilities)
            data_dict['question'].append(prompt)
            data_dict['correct_answer'].append(expected_answer)
            data_dict['model_precision'].append(model_precision)
    return data_dict

def convert_answer_to_bool(answer:str) -> bool|float:
    """
    Converts the answer to a boolean value or NaN.

    :param answer: String of model answer to a yes/no question.
    :return: True if answer starts with 'j' or 'J',
             False if it starts with 'n' or 'N',
             otherwise True if 'ja' is in the lower answer or False if not.
    """
    answer = re.sub(r'^[^a-zA-Z0-9]+', '', answer)
    if answer.lower().startswith('j'):
        return True
    elif answer.lower().startswith('n'):
        return False
    else:
        if 'ja' in answer.lower():
            return True
        else:
            return False

def evaluate_models(df:pd.DataFrame, model_list:list[str], include_baseline:bool=True):
    """
    Evaluate classification performance metrics for a list of models using a DataFrame of predictions.

    For each model in the provided list, this function computes:
      - Accuracy
      - Balanced accuracy
      - Cohen's kappa score

    Optionally, it also evaluates a baseline model that always predicts `False`, using the same
    subset of data as the first model in `model_list`.

    :param df: A pandas DataFrame containing model predictions. Must include the columns:
               - 'model': model identifier
               - 'correct_bool_answer': ground truth boolean labels
               - 'bool_answer': model's predicted boolean labels
    :param model_list: A list of model names (as strings) to evaluate.
    :param include_baseline: If True, evaluates a baseline model that always predicts False.
    :return: A dictionary where keys are model names and values are dictionaries containing:
             - 'accuracy': standard accuracy score
             - 'balanced_accuracy': balanced accuracy score
             - 'cohen_kappa': Cohen's kappa score
    """
    results = {}

    for model_name in model_list:
        model_df = df[df['model'] == model_name]
        y_true = model_df['correct_bool_answer']
        y_pred = model_df['bool_answer']

        acc = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        results[model_name] = {
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "cohen_kappa": kappa
        }

    if include_baseline:
        baseline_df = df[df['model'] == model_list[0]].copy()
        baseline_df['bool_answer'] = False
        y_true = baseline_df['correct_bool_answer']
        y_pred = baseline_df['bool_answer']

        acc = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        results["always_false_baseline"] = {
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "cohen_kappa": kappa
        }

    return results


def compute_distribution_divergences(
    df: pd.DataFrame,
    model_list: list[str],
    unquantized_model: str
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Computes the KL divergence and Hellinger distance between the probability distributions
    of quantized models and a reference unquantized model.

    :param df: DataFrame containing a 'model' column and a 'probabilities' column (each entry is a tensor).
    :param model_list: List of model names to compare.
    :param unquantized_model: Name of the reference (unquantized) model.
    :return: Tuple of two dictionaries:
             - KL divergence results: {model_name: mean_kl_divergence}
             - Hellinger distance results: {model_name: mean_hellinger_distance}
    """
    ref_probs = df[df['model'] == unquantized_model]['probabilities'].to_list()
    ref_probs = torch.stack(ref_probs, dim=-1).detach().cpu().float().squeeze().numpy()

    kl_results = {}
    hellinger_results = {}

    for model_name in model_list:
        if model_name == unquantized_model:
            continue

        data_subset = df[df['model'] == model_name]
        comp_probs = data_subset['probabilities'].to_list()
        comp_probs = torch.stack(comp_probs, dim=-1).detach().cpu().float().squeeze().numpy()

        if ref_probs.shape != comp_probs.shape:
            print(f"Shape mismatch between {unquantized_model} and {model_name}")
            continue

        model_kl_divs = np.sum(kl_div(comp_probs, ref_probs), axis=0)
        sqrt_diff = np.sqrt(comp_probs) - np.sqrt(ref_probs)
        hellinger_distance = np.sqrt(np.sum(sqrt_diff ** 2, axis=0)) / np.sqrt(2)

        kl_results[model_name] = np.mean(model_kl_divs)
        hellinger_results[model_name] = np.mean(hellinger_distance)

    return kl_results, hellinger_results
