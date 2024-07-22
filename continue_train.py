import json, os
from typing import Dict
import safetensors

import torch
from utils.arguments import TrainArguments
from utils.data_loader import load_and_preprocess_data
from utils.callbacks import ParamNormCallback
from utils.metrics import compute_metrics

from trl import SFTTrainer
from peft import PeftConfig, get_peft_model, prepare_model_for_kbit_training
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq


def get_bnb_config(config: TrainingArguments) -> Dict:
    file_path = os.path.join(config.checkpoint_dir, "bnb_config.json")
    with open(file_path, 'r') as f:
        bnb_config = json.load(f)
    
    return bnb_config


def get_training_args(checkpoint: str) -> TrainArguments:
    training_args_path = os.path.join(checkpoint, "training_args.bin")
    training_args = torch.load(training_args_path)

    return training_args


def main(config: TrainingArguments):
    checkpoint = os.path.join(config.checkpoint_dir, config.checkpoint)

    try:
        peft_config = PeftConfig.from_pretrained(checkpoint)
        bnb_config = get_bnb_config(config)
        training_args = get_training_args(checkpoint)
        adapter_weights = safetensors.torch.load_file(
            os.path.join(checkpoint, "adapter_model.safetensors")
        )
    except FileNotFoundError as e:
        print(e)

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map={"":0},
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    train_data, valid_data = load_and_preprocess_data(config, tokenizer)

    model = get_peft_model(model, peft_config)

    state_dict = get_peft_model_state_dict(model)
    print(state_dict)
    # model = set_peft_model_state_dict(model, adapter_weights)

    # model.config.use_cache = False
    # model = torch.compile(model)

    # model.print_trainable_parameters()

    # trainer = SFTTrainer(
    #     model,
    #     train_dataset=train_data,
    #     valid_dataset=valid_data,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     callbacks=[ParamNormCallback],
    #     compute_metrics=compute_metrics,
    #     data_collator=DataCollatorForSeq2Seq(
    #         tokenizer, padding=True, return_tensors="pt", pad_to_multiple_of=8
    #     )
    # )

    # trainer.train(resume_training_from_checkpoint=checkpoint)

    # trainer.model.save_pretrained(config.lora_save_dir)

    # eval_result = trainer.evaluate(eval_dataset=valid_data)
    # print(eval_result)




if __name__ == "__main__":
    train_args = TrainArguments.define_args()
    main(train_args)

