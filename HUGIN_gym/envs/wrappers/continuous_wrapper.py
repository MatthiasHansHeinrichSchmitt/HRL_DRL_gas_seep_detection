import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ContinuousToDiscreteActionWrapper(gym.ActionWrapper):
    """
    Wrap a discrete-action env into a continuous-action env for SAC.

    Original: Discrete(n_actions)
    Wrapped:  Box(shape=(1,), low=-1, high=1)

    The scalar in [-1,1] is binned into n_actions discrete actions.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Discrete), \
            "ContinuousToDiscreteActionWrapper requires a Discrete action space."
        self.n_actions = env.action_space.n
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, act_continuous):
        """
        Map continuous scalar in [-1,1] to discrete action {0, ..., n_actions-1}.
        """
        x = float(np.clip(act_continuous[0], -1.0, 1.0))
        # map [-1,1] -> [0, n_actions)
        frac = (x + 1.0) / 2.0  # in [0,1]
        disc = int(frac * self.n_actions)
        disc = min(disc, self.n_actions - 1)
        return disc
