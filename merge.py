import torch
from utils.arguments import TrainArguments
from transformers import AutoModelForCausalLM
from peft import PeftModel


def main(train_args):
    base_model = AutoModelForCausalLM.from_pretrained(
        train_args.base_model,
        device_map="auto",
        torch_dtype=torch.float16
    )

    merged_model = PeftModel.from_pretrained(
        base_model,
        train_args.lora_save_dir,
        device_map="auto",
        torch_dtype=torch.float16
    )

    merged_model.save_pretrained(train_args.merged_model_dir)
    print(merged_model)


if __name__ == "__main__":
    train_args = TrainArguments.define_args()
    main(train_args)