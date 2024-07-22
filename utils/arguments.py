from typing import List
import argparse


class TrainArguments:
    eeve = "yanolja/EEVE-Korean-Instruct-2.8B-v1.0"
    kullm = "nlpai-lab/kullm-polyglot-5.8b-v2"

    @classmethod
    def define_args(cls):
        p = argparse.ArgumentParser()

        # model/data parameters
        p.add_argument(
            "--base_model", type=str, default="eeve",
            help="Select a model you want to train. e.g) --base_model kullm, default=eeve"
            )
        p.add_argument("--data_path", type=str, default="Smoked-Salmon-s/empathetic_dialogues_ko")
        p.add_argument("--prompt_template_name", type=str, default="multi")
        p.add_argument("--merged_model_dir", type=str, default="./merged_model")
        p.add_argument("--add_eos_token", action="store_true")
        p.add_argument("--num_toys", type=int, default=0, help="Using few datasets for validation of all processes.")
        p.add_argument("--verbose", action="store_true", help="Print templated data.")
        # training hyperparameters
        p.add_argument("--max_seq_len", type=int, default=512)
        p.add_argument("--valid_ratio", type=float, default=0.2)
        p.add_argument("--per_device_train_batch_size", type=int, default=4)
        p.add_argument("--per_device_eval_batch_size", type=int, default=2)
        p.add_argument("--gradient_accumulation_steps", type=int, default=16)
        p.add_argument("--eval_accumulation_steps", type=int, default=16)
        p.add_argument("--num_epochs", type=int, default=3)
        p.add_argument("--max_steps", type=int, default=0)
        p.add_argument("--learning_rate", type=float, default=3e-4)
        p.add_argument("--save_total_limit", type=int, default=5)
        p.add_argument("--logging_dir", type=str, default="./logs")
        p.add_argument("--logging_steps", type=int, default=200)
        p.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
        p.add_argument("--eval_steps", type=int, default=200)
        p.add_argument("--save_steps", type=int, default=400)
        p.add_argument("--weight_decay", type=float, default=0.01)
        p.add_argument("--warmup_ratio", type=float, default=0.1)
        p.add_argument("--lr_scheduler_type", type=str, default="linear")
        p.add_argument("--use_compute_metrics", action="store_true")
        p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory where checkpoints are saved. default='/checkpoints'")
        p.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint to start continue training.")
        p.add_argument("--save_model_weights", action="store_true", 
                       help="Save the LoRA adapter model weights at each checkpoint.")
        # lora hyperparameters
        p.add_argument("--lora_rank", type=int, default=8)
        p.add_argument("--lora_alpha", type=int, default=32)
        p.add_argument("--lora_dropout", type=float, default=0.)
        p.add_argument("--lora_save_dir", type=str, default="./lora_adapter")
        
        args = p.parse_args()
        if args.base_model == "eeve":
            args.base_model = cls.eeve
        elif args.base_model == "kullm":
            args.base_model = cls.kullm
        else:
            raise ValueError(f"Unsupport base model: {args.base_model}, choose between eeve or kullm.")

        return args