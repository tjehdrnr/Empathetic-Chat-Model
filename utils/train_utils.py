import os, re
from typing import List
from argparse import ArgumentParser
from transformers import AutoTokenizer
from peft import PeftModel


def print_trainable_parameters(model) -> None:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || All params: {all_params} Trainable%: {100 * trainable_params // all_params}"
    )


def resume_from_checkpoint(train_args: ArgumentParser):
    checkpoint_path = os.path.join(
            train_args.checkpoint_dir, train_args.checkpoint
        )
    checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)

    if train_args.checkpoint == "":
        raise Exception("You must specify a checkpoint. e.g. --checkpoint checkpoint-500")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Please check your checkpoint dir path. You specified path as '{checkpoint_path}'")
    
    model = PeftModel.from_pretrained(checkpoint_path, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    print(f"Resume model training from {train_args.checkpoint}")
    print(model)
    print(tokenizer)

    return model, tokenizer

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