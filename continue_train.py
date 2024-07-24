import json, os
from typing import Dict
import safetensors
import argparse

import torch
from utils.arguments import Arguments
from utils.data_loader import load_and_preprocess_data
from utils.callbacks import ParamNormCallback, SavePeftModelCallback
from utils.metrics import compute_metrics
from utils.train_utils import get_parsed_arguments

from trl import SFTTrainer
from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq


def get_bnb_config(config: TrainingArguments) -> Dict:
    file_path = os.path.join(config.checkpoint_dir, "bnb_config.json")
    with open(file_path, 'r') as f:
        bnb_config = json.load(f)
    
    return bnb_config


def get_training_args(checkpoint: str) -> TrainingArguments:
    training_args_path = os.path.join(checkpoint, "training_args.bin")
    training_args = torch.load(training_args_path)

    return training_args


def main(config: TrainingArguments):
    """
    If you want to resume training, you must have safetensor weight file in the checkpoint directory.
    When training, you can save this file with the '--save_model_weights' option.
    Once you have your weight file ready, run it as follows:
        e.g. python continue_train.py --checkpoint checkpoint-200
    You only need to specify a checkpoint at which to resume training.
    """
    checkpoint = os.path.join(config.checkpoint_dir, config.checkpoint)
    if not os.path.exists(config.checkpoint_dir):
        raise FileNotFoundError(f"Can not find '{config.checkpoint_dir}' directory")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Can not find '{config.checkpoint}' at '{config.checkpoint_dir}' directory")

    try:
        peft_config = PeftConfig.from_pretrained(checkpoint)
        bnb_config = get_bnb_config(config)
        training_args = get_training_args(checkpoint)
        adapter_weights = safetensors.torch.load_file(
            os.path.join(checkpoint, "adapter_model.safetensors")
        )
    except FileNotFoundError as e:
        print(e)

    peft_config.inference_mode = False

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map={"":0},
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if "EEVE" in config.base_model:
        model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    train_data, valid_data = load_and_preprocess_data(config, tokenizer)

    model = get_peft_model(model, peft_config)
    model.config.use_cache = False


    set_peft_model_state_dict(model, adapter_weights)

    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[SavePeftModelCallback],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, return_tensors="pt", pad_to_multiple_of=8
        )
    )
    print(':'*30 + f"Training resumes from {config.checkpoint}" + ':'*30)
    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.model.save_pretrained(config.lora_save_dir)

    if valid_data is not None:
        eval_result = trainer.evaluate(eval_dataset=valid_data)
        print(eval_result)



if __name__ == "__main__":
    config = Arguments.define_train_args()

    # Load arguments used in previous training.
    config = get_parsed_arguments(config)

    main(config)

