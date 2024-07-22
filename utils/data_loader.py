import warnings
from typing import Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from argparse import ArgumentParser

from utils.arguments import TrainArguments
from utils.prompter import Prompter


def load_and_preprocess_data(train_args: ArgumentParser, tokenizer: AutoTokenizer) -> DatasetDict:
    try:
        dataset = load_dataset(train_args.data_path)
    except ValueError:
        print(f"Your dataset path sets {train_args.data_path}, please check it.")
    
    # Using for validation of all processes only.
    if train_args.num_toys > 0:
        warnings.warn(
            f"You are using a toy dataset of size {train_args.num_toys}. If you do not want this, set '--num_toys' to 0.",
            UserWarning
        )
        dataset = Dataset.from_dict(dataset['train'][:train_args.num_toys])
        dataset = DatasetDict({'train': dataset})
    
    prompter = Prompter(template_name=train_args.prompt_template_name, verbose=train_args.verbose)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=train_args.max_seq_len,
            padding=False,
            return_tensors=None,
        )
        if(
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < train_args.max_seq_len
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
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=train_args.add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"]) - int(train_args.add_eos_token)
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt
    
    # For remove unnecessary columns.
    original_fields = dataset['train'].column_names
    if train_args.valid_ratio > 0:
        tandv = dataset['train'].train_test_split(test_size=train_args.valid_ratio, shuffle=True, seed=42)
        train_data = tandv['train'].map(generate_and_tokenize_prompt, remove_columns=original_fields)
        valid_data = tandv['test'].map(generate_and_tokenize_prompt, remove_columns=original_fields)
    else:
        train_data = dataset['train'].map(generate_and_tokenize_prompt, remove_columns=original_fields)
        valid_data = None
    
    return train_data, valid_data