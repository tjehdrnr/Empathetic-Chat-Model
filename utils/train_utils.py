import os, re
import json
from typing import List
from types import SimpleNamespace
from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


def print_trainable_parameters(model) -> None:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || All params: {all_params} Trainable %: {100 * trainable_params / all_params:.4f}"
    )


def get_parsed_arguments(config: ArgumentParser):
    """
    This function loads the previously trained arguments. 
    If you want to train with a new arguments, you should not use this function.
    """
    saved_config = os.path.join(config.checkpoint_dir, "parsed_args.json")
    new_config = None
    try:
        with open(saved_config, 'r') as f:
            new_config = json.load(f)
    except FileNotFoundError as e:
        print(e)

    new_config = SimpleNamespace(**saved_config)
    
    if config.checkpoint is not None:
        new_config.checkpoint = config.checkpoint
    else:
        raise ValueError(
            """You must specify a checkpoint at which to resume training. 
            e.g) python continue_train.py --checkpoint checkpoint-500"""
        )

    return new_config


# Get lora trainable target modules from model.
def get_target_modules(model) -> List[str]:
    model_modules = str(model.modules)
    linear_layers = re.findall(r"\((\w+)\): Linear", model_modules)
    
    target_modules = []
    for name in linear_layers:
        if name != "lm_head": # needed for 16-bit
            target_modules.append(name)
    target_modules = list(set(target_modules))
    print(f"LoRA target module list: {target_modules}")

    return target_modules