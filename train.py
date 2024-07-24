import json, os
from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from utils.arguments import Arguments
from utils.data_loader import load_and_preprocess_data
from utils.train_utils import *
from utils.callbacks import ParamNormCallback, SavePeftModelCallback
from utils.metrics import compute_metrics


def train(config: ArgumentParser):  
    # Load tokenizer and set custormized options.
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.padding_side = "left" # allow batched inference
    # """ 
    #  Replace pad_token with a new unk_token for left padding.
    #  This process causes label tokens set to -100 to be ignored
    #  during the calculation process.
    # """
    if "EEVE" in config.base_model:
        tokenizer.add_special_tokens({"pad_token": "<|unused|>"})
        new_pad_token_id = tokenizer.convert_tokens_to_ids("<|unused|>")
        tokenizer.pad_token_id = new_pad_token_id # Use the newly added unk token
        assert tokenizer.pad_token_id == new_pad_token_id, f"pad_token_id is not set to {new_pad_token_id}"
    else: # kullm model
        tokenizer.pad_token_id = 0 # Use an already existing unk token
        assert tokenizer.pad_token_id == 0, "pad_token_id is not set to 0"
    print(f"pad_token: {tokenizer.pad_token} pad_token_id: {tokenizer.pad_token_id}")

    # Load preprocessed data.
    train_dataset, eval_dataset = load_and_preprocess_data(config, tokenizer)

    # Define bitsandbytes configurations and load model.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    bnb_config.to_json_file(
        os.path.join(config.output_dir, "bnb_config.json")
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map={"":0},
    )

    # If you added new tokens to the tokenizer, resize the vocab dim of embedding.
    if "EEVE" in config.base_model:
        model.resize_token_embeddings(len(tokenizer))

    # Improve memory efficiency and save GPU memory.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # Define lora configurations and apply lora to model.
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=get_target_modules(model),
        task_type="CAUSAL_LM",
        bias="none",
    )
    # A model that combines model with LoRA.
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    # print_trainable_parameters(model)
    model.print_trainable_parameters()

    # Define training arguments and train.
    training_args = TrainingArguments(
        # overwrite_output_dir=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=config.eval_accumulation_steps if config.do_eval else None,
        optim=config.optimizer,
        eval_strategy="steps" if config.do_eval else "no",
        eval_steps=config.eval_steps if config.do_eval else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        output_dir=config.output_dir,
        save_total_limit=config.save_total_limit,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        fp16=True, # Use mixed precision, same as using 'torch.autocast()'.
        load_best_model_at_end=True if config.do_eval else False,
        save_safetensors=config.save_safetensors,
        # group_by_length=True,
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[SavePeftModelCallback],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, return_tensors="pt", pad_to_multiple_of=8
        ),
    )

    trainer.train()

    # Save lora adapter model.
    trainer.model.save_pretrained(config.lora_save_dir)

    if eval_dataset is not None:
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        print(eval_result)



def main():
    config = Arguments.define_args()
    
    if not os.path.exits(config.output_dir):
        os.mkdir(config.output_dir)

    parsed_config = os.path.join(config.output_dir, "parsed_args.json")
    with open(parsed_config, 'w') as f:
        json.dump(vars(config), f)

    train(config)



if __name__ == "__main__":
    main()