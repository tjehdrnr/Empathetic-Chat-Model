from transformers import TrainerCallback
from utils.metrics import compute_perlexity


class OnStepEnd(TrainerCallback):

    @staticmethod
    def get_parameter_norm(model):
         # compute parameter norm
        total_param_norm = 0
        try:
            for p in model.parameters():
                param_norm = p.data.norm(2)
                total_param_norm += param_norm.item() ** 2
            total_param_norm = total_param_norm ** (1. / 2)
        except Exception as e:
            print(e)
        
        return total_param_norm


    def on_step_end(self, args, state, control, **kwargs):
        """
        Event called after logging the last logs.
        """
        model = kwargs['model']

        logits = getattr(model, 'logits', None)
        labels = getattr(model, 'labels', None)

        if logits is not None and labels is not None:
            eval_preds = (logits, labels)
            result = compute_perlexity(eval_preds)
            ppl = result['ppl']

        total_param_norm = self.get_parameter_norm(model)
        
        print(f"Step: {state.global_step} || Perplexity: {ppl} || Parameter norm: {total_param_norm}")

        return control
