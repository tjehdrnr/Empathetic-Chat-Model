import torch
from utils.arguments import TrainArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main(train_args):
    base_model = AutoModelForCausalLM.from_pretrained(
        train_args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Matches the tokenizer of the lora adapter and the base model.
    tokenizer = AutoTokenizer.from_pretrained(train_args.base_model, device_map="auto")
    if "EEVE" in train_args.base_model:
        tokenizer.add_special_tokens({"pad_token": "<|unused|>"})
        new_pad_token_id = tokenizer.convert_tokens_to_ids("<|unused|>")
        tokenizer.pad_token_id = new_pad_token_id
        # Resize the embedding dim of the base model.
        base_model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer.pad_token_id = 0 # unk token

    # Apply LoRA to the base model.
    merged_model = PeftModel.from_pretrained(
        base_model,
        train_args.lora_save_dir,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Merge base model and LoRA adapter.
    merged_model.merge_and_unload()

    merged_model.save_pretrained(train_args.merge_dir)
    tokenizer.save_pretrained(train_args.merge_dir)
    print("The merge process has been completed.")
    print(merged_model)



if __name__ == "__main__":
    train_args = TrainArguments.define_args()
    main(train_args)