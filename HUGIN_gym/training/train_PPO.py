# train_PPO.py
import gymnasium as gym
from stable_baselines3 import PPO
import pickle  # saving the stats
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from HUGIN_gym.utils.io import confirm_overwrite
from HUGIN_gym.envs.wrappers.DynamicEpisodeLengthWrapper import DynamicEpisodeLengthWrapper
from HUGIN_gym.envs.wrappers.FilterObservationWrapper import FilterObservationWrapper
from HUGIN_gym.callbacks.EpisodeStatsCallback import EpisodeStatsCallback
from HUGIN_gym.utils.build_train_config import build_training_config_ppo
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper


def main():

    MAX_EPS_LEN = 8000
    NUM_ENVS = 12
    GP =True
    ACCURACY_GOALS = [0.90, 1.0, 1.0] #0.9
    SUB_AGENT_TRAIN_ON_GP = True
    AGENT_TYPE = "SPACE"

    if AGENT_TYPE == "BORDER":
        from HUGIN_gym.agents.feature_extractor.agent3_border_explore import (
            AgentBoarderExplore as Agent,
        )
        keys_you_want_to_keep = [
            "obs_state",
            #"visited_maps_downsampled",
            #"GT_c_around_threshold_maps_downsampled",
            "local_GT_around",
            "border_coverage",
            #"space_coverage",
            "distance_from_max",
        ]
        agent_reward_path = "./HUGIN_gym/envs/core/rewards/agent3_border.py"

    elif AGENT_TYPE == "SPACE":
        from HUGIN_gym.agents.feature_extractor.agent1_space_explore import (
            AgentSpaceExplore as Agent,
        )
        keys_you_want_to_keep = [
            "obs_state",
            "visited_maps_downsampled",
            "space_coverage",
            #"VAR_visited_downsampled"
        ]
        agent_reward_path = "./HUGIN_gym/envs/core/rewards/agent1_exploration.py"

    elif AGENT_TYPE == "PLUME":
        from HUGIN_gym.agents.feature_extractor.agent2_plume_explore import (
            AgentPlumeExplore as Agent,
        )
        keys_you_want_to_keep = [
            "obs_state",
            #"visited_maps_downsampled",
            #"GT_c_over_threshold_maps_downsampled",
            "local_GT",
            "plume_coverage",
            #"space_coverage",
            "distance_from_max",
        ]
        agent_reward_path = "./HUGIN_gym/envs/core/rewards/agent2_plume.py"

    NAME = "3D_PPO_border_GP_sub_task_completion_41x41x41_7k_GP"#spawn_close_to_source
    saving_location = f"../trained-agents/{NAME}"
    loading_location = "../trained-agents/..._what_so_ever_..."

    confirm_overwrite(saving_location)

    def episode_length_schedule(local_step):
        # Placeholder for a more complex function
        return MAX_EPS_LEN

    # GP init
    kernel_config = {
        "RBF": {"type": "RBF", "length_scale": 3.5},
    }

    def make_env():
        def _init():
            env = gym.make("HUGIN-v0")
            env = env.unwrapped  # remove default TimeLimit
            env.train = True
            env.accuracy_agent_goals = ACCURACY_GOALS
            env.GP_ON = GP
            env.multiple_gaussians = [1,1]
            #env.spawn_close_to_source = True # # !!!!!!!!!!!!!!!!!!'#####!!!!!!!!!!!!!!!!!!!
            if AGENT_TYPE == "SPACE":
                env.use_c_map = False
            else:
                env.use_c_map = True
            env.agent_type = AGENT_TYPE

            env = DynamicEpisodeLengthWrapper(
                env, schedule_fn=episode_length_schedule
            )

            if GP:
                gp_env = GPWrapper(
                    env,
                    kernels_config=kernel_config,
                    HRL=False,
                    SUB_AGENT_TRAIN_ON_GP=SUB_AGENT_TRAIN_ON_GP,
                )
                gp_env.accuracy_goals = ACCURACY_GOALS
                gp_env.reset()
            else:
                gp_env = env

            gp_env = FilterObservationWrapper(
                gp_env, keys_to_keep=keys_you_want_to_keep
            )
            return gp_env

        return _init

    # get max_return / N_STATES for stats callback
    dummy_env = make_env()()
    N_STATES = dummy_env.unwrapped.max_return
    dummy_env.close()

    env_fns = [make_env() for _ in range(NUM_ENVS)]
    env = SubprocVecEnv(env_fns)

    policy_kwargs = dict(features_extractor_class=Agent)

    load_existing = False  # switch to True if you want to resume

    if load_existing:
        model = PPO.load(
            f"{loading_location}/PPO_scratch", env=env, tensorboard_log="./tensorboard_logs/"
        )
    else:
        # PPO hyperparameters – you can tune these
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,               # per env; effective batch size = n_steps * NUM_ENVS
            batch_size=1024,             # must divide n_steps * NUM_ENVS in SB3, but SB3 will handle remainder
            n_epochs=4, #4 for longer episodes
            gamma=0.9997,                # keep your discount for comparability
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            policy_kwargs=policy_kwargs,
            # tensorboard_log="./tensorboard_logs/",
        )

    max_steps = 100_000_000
    # For PPO, save every X *env steps*; same formula as before:
    checkpoint_steps = 10_500_020 // NUM_ENVS

    stats_callback = EpisodeStatsCallback(
        max_episode_length=MAX_EPS_LEN,
        save_freq=checkpoint_steps,
        NUM_ENVS=NUM_ENVS,
        address=f"{saving_location}/episode_stats",
        N_states=N_STATES,
        GP=GP,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_steps,
        save_path=f"{saving_location}/",
        name_prefix="PPO_checkpoint",
        save_replay_buffer=False,  # PPO has no replay buffer, but flag is accepted
    )

    # Save training config
    build_training_config_ppo(
        model,
        AGENT_TYPE,
        keys_you_want_to_keep,
        agent_reward_path,
        max_steps,
        MAX_EPS_LEN,
        checkpoint_steps,
        saving_location,
        kernel_config=kernel_config if GP else None,
    )

    # TRAIN
    model.learn(
        total_timesteps=max_steps,
        callback=[stats_callback, checkpoint_callback],
        progress_bar=True,
    )

    # SAVING
    stats = stats_callback.get_stats()
    with open(f"{saving_location}/training_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    model.save(f"{saving_location}/PPO_scratch")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
