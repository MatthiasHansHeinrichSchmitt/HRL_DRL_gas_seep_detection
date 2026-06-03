import gymnasium as gym

class DynamicEpisodeLengthWrapper(gym.Wrapper):
    def __init__(self, env, schedule_fn):
        super().__init__(env)
        self.schedule_fn = schedule_fn
        self.current_step = 0
        self.max_episode_steps = self.schedule_fn(self.current_step)
        self.elapsed_steps = 0

    def reset(self, **kwargs):
        self.max_episode_steps = self.schedule_fn(self.current_step)
        self.elapsed_steps = 0
        #print(f"[Env {id(self)}] Reset at local step {self.current_step} → max_episode_steps: {self.max_episode_steps}")
        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_step += 1
        self.elapsed_steps += 1
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.elapsed_steps >= self.max_episode_steps:
            truncated = True

        return obs, reward, terminated, truncated, info


