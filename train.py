import os, re
from typing import List
from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from utils.arguments import TrainArguments
from utils.data_loader import load_and_preprocess_data
from utils.train_utils import *
from utils.callbacks import ParamNormCallback
from utils.metrics import compute_metrics



def train(train_args: ArgumentParser):  
    # Load tokenizer and set custormized options.
    tokenizer = AutoTokenizer.from_pretrained(train_args.base_model)
    tokenizer.padding_side = "left" # allow batched inference
    # """ 
    #  Replace pad_token with a new unk_token for left padding.
    #  This process causes label tokens set to -100 to be ignored
    #  during the calculation process.
    # """
    if "EEVE" in train_args.base_model:
        tokenizer.add_special_tokens({"pad_token": "<|unused|>"})
        new_pad_token_id = tokenizer.convert_tokens_to_ids("<|unused|>")
        tokenizer.pad_token_id = new_pad_token_id # Use the newly added unk token
        assert tokenizer.pad_token_id == new_pad_token_id, f"pad_token_id is not set to {new_pad_token_id}"
    else:
        tokenizer.pad_token_id = 0 # Use an already existing unk token
        assert tokenizer.pad_token_id == 0, "pad_token_id is not set to 0"
    print(f"pad_token: {tokenizer.pad_token} pad_token_id: {tokenizer.pad_token_id}")

    # Load and preprocess data.
    train_data, valid_data = load_and_preprocess_data(train_args, tokenizer)

    # Define bitsandbytes configurations and load model.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        train_args.base_model,
        quantization_config=bnb_config,
        device_map={"":0}
    )
    model = AutoModelForCausalLM.from_pretrained(
        train_args.base_model,
        device_map="cpu"
    )
    # If you added new tokens to the tokenizer, resize the vocab dim of embedding.
    if "EEVE" in train_args.base_model:
        model.resize_token_embeddings(len(tokenizer))

    # Improve memory efficiency and save GPU memory.
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Define lora configurations and apply lora to model.
    lora_config = LoraConfig(
        r=train_args.lora_rank,
        lora_alpha=train_args.lora_alpha,
        lora_dropout=train_args.lora_dropout,
        target_modules=get_target_modules(model),
        task_type="CAUSAL_LM",
        bias="none",
    )
    # A model that combines model with LoRA.
    model = get_peft_model(model, lora_config)

    

    # If you specify this argument, the model resume training from checkpoint.
    if train_args.resume_from_checkpoint:
        model, tokenizer = resume_from_checkpoint(train_args, bnb_config)

    model.config.use_cache = False

    # print_trainable_parameters(model)
    model.print_trainable_parameters()

    # Define training arguments and train.
    training_args = TrainingArguments(
        # overwrite_output_dir=True,
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        per_device_eval_batch_size=train_args.per_device_eval_batch_size,
        num_train_epochs=train_args.num_epochs,
        max_steps=train_args.max_steps,
        learning_rate=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
        warmup_ratio=train_args.warmup_ratio,
        lr_scheduler_type=train_args.lr_scheduler_type,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        eval_accumulation_steps=train_args.eval_accumulation_steps if train_args.valid_ratio > 0 else None,
        optim=train_args.optimizer,
        eval_strategy="steps" if train_args.valid_ratio > 0 else "no",
        eval_steps=train_args.eval_steps if train_args.valid_ratio > 0 else None,
        save_strategy="steps",
        save_steps=train_args.save_steps,
        output_dir=train_args.checkpoint_dir,
        save_total_limit=train_args.save_total_limit,
        logging_dir=train_args.logging_dir,
        logging_steps=train_args.logging_steps,
        fp16=True, # Use mixed precision, same as using 'torch.autocast()'.
        load_best_model_at_end=True if train_args.valid_ratio > 0 else False,        
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[ParamNormCallback],
        compute_metrics=compute_metrics if train_args.use_compute_metrics else None,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, return_tensors="pt", pad_to_multiple_of=8
        ),
    )

    trainer.train()

    # Save lora adapter model.
    trainer.model.save_pretrained(train_args.lora_save_dir)

    eval_result = trainer.evaluate(eval_dataset=valid_data)
    print(eval_result)



def main():
    train_args = TrainArguments.define_args()
    train(train_args)


if __name__ == "__main__":
    main()