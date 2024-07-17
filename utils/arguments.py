from typing import List
import argparse


def train_arguments():
    p = argparse.ArgumentParser()

    # model/data parameters
    p.add_argument("--base_model", type=str, default="yanolja/EEVE-Korean-Instruct-2.8B-v1.0")
    p.add_argument("--data_path", type=str, default="Smoked-Salmon-s/empathetic_dialogues_ko")
    p.add_argument("--prompt_template_name", type=str, default="multi")
    p.add_argument("--merged_model_dir", type=str, default="merged_model")
    p.add_argument("--add_eos_token", type=bool, default=True)
    p.add_argument("--num_toys", type=int, default=0, help="Using few datasets for validation of all processes.")
    p.add_argument("--verbose", action="store_true", help="Print templated data.")
    # training hyperparameters
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--valid_ratio", type=float, default=0.2)
    p.add_argument("--per_device_train_batch", type=int, default=4)
    p.add_argument("--per_device_valid_batch", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument("--logging_dir", type=str, default="./logs")
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--use_compute_metrics", type=bool, default=False)
    p.add_argument(
        "--resume_from_checkpoint", action="store_true",
        help="If you want to continue trainining the model, use this option.\
             And then, specify the checkpoint path."
    )
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint", type=str, default="")
    # lora hyperparameters
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.)
    
    args = p.parse_args()

    return args
    

