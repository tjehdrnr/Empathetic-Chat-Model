from transformers import TrainerCallback


class NormsCallback(TrainerCallback):

    def __init__(self, norm_type: int = 2):
        self.norm_type = norm_type # L2 norm
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Event called at the end of a training step. 
        If using gradient accumulation, one training step might take several inputs.
        """
        model = kwargs['model']
        # Compute parameter norm.
        total_param_norm = 0.0
        try:
            for p in model.parameters():
                param_norm = p.data.norm(self.norm_type)
                total_param_norm += param_norm.item() ** self.norm_type
            total_param_norm = total_param_norm ** (1. / self.norm_type)
        except Exception as e:
            print(e)
        
        # Compute gradient norm.
        total_grad_norm = 0.0
        try:
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(self.norm_type)
                    total_grad_norm += param_norm.item() ** self.norm_type
            total_grad_norm = total_grad_norm ** (1. / self.norm_type)
        except Exception as e:
            print(e)
        
        print(f"Step: {state.global_step} || Gradient Norm: {total_grad_norm:.4f} || Parameter Norm: {total_param_norm:.4f}")

        return control # Optional