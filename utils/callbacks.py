from transformers import TrainerCallback


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
        
        print(f"Step: {state.global_step} || Parameter Norm: {total_param_norm}")

        return control