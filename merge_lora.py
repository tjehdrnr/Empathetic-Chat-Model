import os
import torch
from argparse import ArgumentParser
from utils.arguments import Arguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def get_tokenizer_path(train_args: ArgumentParser) -> str:
    try:
        checkpoint_dns = [
            fn for fn in os.listdir(train_args.output_dir)
            if fn.startswith(PREFIX_CHECKPOINT_DIR)
        ]
    except FileExistsError as e:
        print(e)

    tokenizer_path = os.path.join(
        train_args.output_dir, checkpoint_dns[0]
    )

    return tokenizer_path


def main(train_args: ArgumentParser):
    base_model = AutoModelForCausalLM.from_pretrained(
        train_args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer_path = get_tokenizer_path(train_args)
    
    # Matches the tokenizer of the lora adapter and the base model.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if "EEVE" in train_args.base_model:
        # Resize the embedding dim of the base model.
        base_model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA to the base model.
    model = PeftModel.from_pretrained(
        base_model,
        train_args.lora_save_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Merge base model and LoRA adapter.
    model = model.merge_and_unload()

    model.save_pretrained(train_args.merge_dir)
    tokenizer.save_pretrained(train_args.merge_dir)
    print("The merge process has been completed.")



if __name__ == "__main__":
    train_args = Arguments.define_train_args()

    main(train_args)