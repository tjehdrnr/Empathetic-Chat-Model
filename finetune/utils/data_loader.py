import warnings
from typing import Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from argparse import ArgumentParser

from arguments import Arguments
from utils.prompter import Prompter


def load_and_preprocess_data(config: ArgumentParser, tokenizer: AutoTokenizer) -> DatasetDict:
    try:
        dataset = load_dataset(config.data_path)
    except ValueError:
        print(f"Your dataset path sets {config.data_path}, please check it.")
    
    # Using for validation of all processes only.
    if config.num_toys > 0:
        warnings.warn(
            f"You are using a toy dataset of size {config.num_toys}. If you do not want this, set '--num_toys' to 0.",
            UserWarning
        )
        dataset = Dataset.from_dict(dataset['train'][:config.num_toys])
        dataset = DatasetDict({'train': dataset})
    
    prompter = Prompter(template_name=config.prompt_template_name, verbose=config.verbose)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=config.max_seq_len,
            padding=False,
            return_tensors=None,
        )
        if(
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < config.max_seq_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            instruction=data_point['instruction'],
            label=data_point['output'],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        user_prompt = prompter.generate_prompt(data_point['instruction'])
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=config.add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"]) - int(config.add_eos_token)
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt
    
    if config.valid_ratio > 0:
        tande = dataset['train'].train_test_split(test_size=config.valid_ratio, shuffle=True, seed=42)
        train_dataset = tande['train'].map(generate_and_tokenize_prompt)
        eval_dataset = tande['test'].map(generate_and_tokenize_prompt)
    else:
        train_dataset = dataset['train'].map(generate_and_tokenize_prompt)
        eval_dataset = None
    
    return train_dataset, eval_dataset