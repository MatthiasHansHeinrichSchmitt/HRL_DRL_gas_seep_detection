from gymnasium.envs.registration import register

register(
    id="HUGIN-v0",
    entry_point="HUGIN_gym.envs.HUGIN_env:HUGIN",  # Note the updated entry_point
    max_episode_steps=100,
)
