# train_HRL_PPO.py

import time
import os
import pickle

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from HUGIN_gym.agents.options import Option
from HUGIN_gym.envs.HRL_env import HierarchicalEnv
from HUGIN_gym.agents.feature_extractor.agent0_HRL import AgentHRLExplore

from HUGIN_gym.callbacks.EpisodeStatsCallback import EpisodeStatsCallback
from HUGIN_gym.envs.wrappers.DynamicEpisodeLengthWrapper import DynamicEpisodeLengthWrapper
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper

from HUGIN_gym.callbacks.EpisodeStatsCallback import EpisodeStatsCallback
from HUGIN_gym.agents.term_fn.termination_fn_agent1 import termination_fn1
from HUGIN_gym.agents.term_fn.termination_fn_agent2 import termination_fn2

from HUGIN_gym.utils.io import confirm_overwrite
from HUGIN_gym.utils.build_train_config import (
    build_training_config_ppo,
    build_subpolicy_metadata,
)


def main():

    NAME = "HRL_PPO_GP_base_agents_GP_1G_term_0p1_uncertainty_AND_all_plume_covered_3_OPT_len_90prosent_d_base_agents_only_subtask_termination_no_space_coverage_and_xyz_right_feature_extractor"
    saving_location = f"../trained-agents/{NAME}"
    loading_location = "../trained-agents/..._what_so_ever_..."

    confirm_overwrite(saving_location)

    # ----- HRL / env configuration -----
    AGENT_TYPE = "META"
    MAX_EPS_LEN = 160
    NUM_ENVS = 12

    GP = True
    MULTIPLE_GAUSSIANS = [1, 1]
    ACCURACY_GOALS = [0.9, 0.9, 0.9]
    MAX_OPT_LENGTH = 3

    kernel_config = {
        "RBF": {"type": "RBF", "length_scale": 3.5},
        # "Matern": {"type": "Matern", "length_scale": 3.5, "nu": 1.5},
    }

    def episode_length_schedule(local_step):
        # Placeholder for more complex logic
        return MAX_EPS_LEN

    # ----- Sub-policies (still DQN) -----
    sub_policies = [
        {
            "id": "option1",
            "name": "spatial exploration",
            "dir": "../trained-agents/PPO_space_GP_without_VAR",
            "model_file": "PPO_scratch.zip",
            "termination_fn": termination_fn1,
            "obs_keys": [
                "obs_state",
                "visited_maps_downsampled",
                "space_coverage",
                #"VAR_visited_downsampled",
            ],
        },
        {
            "id": "option2",
            "name": "gas plume exploration",
            "dir": "../trained-agents/2D_PPO_plume_GP_sub_task_completion_41x41x1",#PPO_plume_GP_clipped_RWD_no_distance_to_max
            "model_file": "PPO_checkpoint_21000024_steps.zip",
            "termination_fn": termination_fn2,
            "obs_keys": [
                "obs_state",
                #"visited_maps_downsampled",
                #"GT_c_over_threshold_maps_downsampled",
                "local_GT",
                "plume_coverage",
                #"space_coverage",
                "distance_from_max",
            ],
        },
        {
            "id": "option3",
            "name": "gas border exploration",
            "dir": "../trained-agents/2D_PPO_border_GP_sub_task_completion_41x41x1",#PPO_border_GP_not_clipped_RWD_no_distance_to_max
            "model_file": "PPO_checkpoint_21000024_steps.zip",
            "termination_fn": termination_fn2,
            "obs_keys": [
                "obs_state",
                #"visited_maps_downsampled",
                #"GT_c_around_threshold_maps_downsampled",
                "local_GT_around",
                "border_coverage",
                #"space_coverage",
                "distance_from_max",
            ],
        },
    ]

    HRL_sub_policy_data = build_subpolicy_metadata(sub_policies)

    # ----- Env factory -----
    def make_env():
        def _init():
            base_env = gym.make("HUGIN-v0")
            base_env = base_env.unwrapped  # REMOVE default TimeLimit wrapper

            # Base env config
            base_env.use_c_map = True
            base_env.GP_ON = GP
            base_env.train = True
            base_env.HRL = True
            base_env.agent_type = AGENT_TYPE
            base_env.multiple_gaussians = MULTIPLE_GAUSSIANS

            # Load sub-policies as Options
            options = []
            for sp in sub_policies:
                model_path = os.path.join(sp["dir"], sp["model_file"])
                policy = PPO.load(model_path)

                option = Option(
                    policy,
                    sp["termination_fn"],
                    sp["name"],
                    obs_keys=sp["obs_keys"],
                )
                options.append(option)

            # Optional GP wrapper around base env
            if GP:
                gp_env = GPWrapper(
                    base_env,
                    kernels_config=kernel_config,
                    gp_visualise=False,
                    HRL=True,
                    SUB_AGENT_TRAIN_ON_GP=False,  # HRL-level training only
                )
            else:
                gp_env = base_env

            # Hierarchical environment with options
            env = HierarchicalEnv(
                base_env=gp_env,
                options=options,
                gamma=0.99,
                MAX_OPT_LENGTH=MAX_OPT_LENGTH,
                ACCURACY_GOALS=ACCURACY_GOALS,
                GP=GP,
            )

            # Control episode length
            env = DynamicEpisodeLengthWrapper(
                env,
                schedule_fn=episode_length_schedule,
            )

            return env

        return _init

    # ----- Extract some metadata (states, keys) -----
    # Create one temp env to extract shape safely
    temp_env = make_env()()
    # For HRL we still read from the underlying base env
    keys_you_want_to_keep = temp_env.env.base_env.observation_space.keys
    N_STATES = temp_env.env.base_env.unwrapped.max_return
    temp_env.close()

    # ----- Vectorized env -----
    env_fns = [make_env() for _ in range(NUM_ENVS)]
    env = SubprocVecEnv(env_fns)

    # ----- PPO meta-controller -----
    policy_kwargs = dict(
        features_extractor_class=AgentHRLExplore,
    )

    load_existing = False  # switch to True to resume PPO meta-controller

    if load_existing:
        model = PPO.load(
            f"{loading_location}/HRL_PPO_scratch",
            env=env,
            tensorboard_log="./tensorboard_logs/",
        )
    else:
        model = PPO(
            "MultiInputPolicy",  # dict observation space
            env,
            learning_rate=3e-4,
            n_steps=512,          # per env
            batch_size=1024,      # effective batch = n_steps * NUM_ENVS
            n_epochs=4,
            gamma=0.9945,         # similar to your PPO single-agent for consistency
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tensorboard_logs/",
        )

    # ----- Training setup & callbacks -----
    max_steps = 10_000_000  # adapt if needed
    checkpoint_steps = 2_050_000 // NUM_ENVS  # same formula as before if you want

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
        name_prefix="HRL_PPO_checkpoint",
        save_replay_buffer=False,  # PPO has no replay buffer
    )

    # ----- Save training config (PPO version) -----
    build_training_config_ppo(
        model,
        AGENT_TYPE,
        keys_you_want_to_keep,
        agent_reward_path=None,  # HRL-level reward is likely internal to HierarchicalEnv
        max_steps=max_steps,
        max_eps_len=MAX_EPS_LEN,
        checkpoint_steps=checkpoint_steps,
        saving_location=saving_location,
        kernel_config=kernel_config if GP else None,
        MAX_OPT_LENGTH=MAX_OPT_LENGTH,
        MULTIPLE_GAUSSIANS=MULTIPLE_GAUSSIANS,
        sub_policy_metadata=HRL_sub_policy_data,
    )

    # ----- Train meta-controller -----
    model.learn(
        total_timesteps=max_steps,
        callback=[stats_callback, checkpoint_callback],
        progress_bar=True,
    )

    # ----- Save stats and model -----
    stats = stats_callback.get_stats()
    # Optional: add waypoints if you still want them
    stats["waypoints"] = list(range(0, model.num_timesteps + 1, checkpoint_steps))
    with open(f"{saving_location}/training_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    model.save(f"{saving_location}/HRL_PPO_scratch")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
