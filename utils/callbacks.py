import os
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class ParamNormCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        """
        Event called after logging the last logs.
        """
        model = kwargs['model']

        # compute parameter norm
        total_param_norm = 0
        try:
            for p in model.parameters():
                param_norm = p.data.norm(2)
                total_param_norm += param_norm.item() ** 2
            total_param_norm = total_param_norm ** (1. / 2)
        except Exception as e:
            print(e)
        
        print(f"Step: {state.global_step} || Parameter Norm: {format(total_param_norm, ',')}")

        return control


class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        """
        Event called after a checkpoint save.
        """
        checkpoint = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        
        return control
    