import gymnasium as gym

class FilterObservationWrapper(gym.ObservationWrapper):
        def __init__(self, env, keys_to_keep):
            super().__init__(env)
            self.keys_to_keep = keys_to_keep

            # Filter obs space to include only requested keys, otherwise it hinders learning and adds noise
            self.observation_space = gym.spaces.Dict({
                k: env.observation_space[k]
                for k in keys_to_keep
            })

        def observation(self, obs):
            return {k: obs[k] for k in self.keys_to_keep}