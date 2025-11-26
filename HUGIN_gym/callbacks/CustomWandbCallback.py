import wandb
from stable_baselines3.common.callbacks import BaseCallback
import torch as th

class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            # Use the last observations from the model
            state = {k: th.tensor(v).float() for k, v in self.model._last_obs.items()}
            q_values = self.model.q_net(state)  # Forward pass to get Q-values
            mean_q = q_values.mean().item()
            wandb.log({"mean_q": mean_q})
        return True
    


