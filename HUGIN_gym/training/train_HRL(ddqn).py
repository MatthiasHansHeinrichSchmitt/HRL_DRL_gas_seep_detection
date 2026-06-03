# train.py
import time


import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from stable_baselines3 import DQN
from HUGIN_gym.agents.options import Option

import pickle # saving the stats
from HUGIN_gym.callbacks.EpisodeStatsCallback import EpisodeStatsCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from HUGIN_gym.envs.HRL_env import HierarchicalEnv
from HUGIN_gym.agents.feature_extractor.agent0_HRL import AgentHRLExplore


from HUGIN_gym.utils.io import confirm_overwrite, save_training_config
from HUGIN_gym.utils.load_reward_function import load_reward_function_source
from HUGIN_gym.envs.wrappers.DynamicEpisodeLengthWrapper import DynamicEpisodeLengthWrapper
from HUGIN_gym.envs.wrappers.FilterObservationWrapper import FilterObservationWrapper
from HUGIN_gym.callbacks.EpisodeStatsCallback import EpisodeStatsCallback
from HUGIN_gym.agents.term_fn.termination_fn_agent1 import termination_fn1
from HUGIN_gym.agents.term_fn.termination_fn_agent2 import termination_fn2
from HUGIN_gym.utils.build_train_config import build_training_config, build_subpolicy_metadata
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper
import os
def main():

    NAME = "HRL_GT_base_agents_GT_not_clipped_1or2G"
    saving_location = f"../trained-agents/{NAME}"
    loading_location = "../trained-agents/..._what_so_ever_..."
    confirm_overwrite(saving_location)
    AGENT_TYPE = "META"
    MAX_EPS_LEN = 130
    NUM_ENVS = 12
    GP=False
    MULTIPLE_GAUSSIANS =[1,2]
    ACCURACY_GOALS = [0.7,0.9,0.9]
    MAX_OPT_LENGTH = 3
    kernel_config = { # REPLACE with a config term?!
                    "RBF": {"type": "RBF", "length_scale": 3.5},}
                    #"Matern": {"type": "Matern", "length_scale": 3.5, "nu": 1.5},}
    def episode_length_schedule(local_step):
        return MAX_EPS_LEN  # Spaceholder for a more complex function
    
    sub_policies = [
            {
                "id": "option1",
                "name": "spatial exploration",
                "dir": "../trained-agents/space_GT_not_clipped_RWD",
                "model_file": "DQN_scratch.zip",
                "termination_fn": termination_fn1,
                "obs_keys": ["obs_state", "visited_maps_downsampled", "space_coverage"],
            },
            {
                "id": "option2",
                "name": "gas plume exploration",
                "dir": "../trained-agents/plume_GT_clipped_RWD",
                "model_file": "DQN_scratch.zip",
                "termination_fn": termination_fn2,
                "obs_keys": ["obs_state","visited_maps_downsampled","GT_c_over_threshold_maps_downsampled","local_GT","plume_coverage","space_coverage"], # modify keys and name !
            },
            {
                "id": "option3",
                "name": "gas border exploration",
                "dir": "../trained-agents/border_GT_not_clipped_RWD",
                "model_file": "DQN_scratch.zip",
                "termination_fn": termination_fn2,
                "obs_keys": ["obs_state","visited_maps_downsampled","GT_c_around_threshold_maps_downsampled","local_GT_around","border_coverage","space_coverage"],# modify keys and name !
            },
        ]
    HRL_sub_policy_data = build_subpolicy_metadata(sub_policies)
    def make_env():
        def _init():
            base_env = gym.make("HUGIN-v0")
            base_env = base_env.unwrapped    # REMOVE default TimeLimit wrapper
            base_env.use_c_map = True # default
            base_env.GP_ON = GP 
            base_env.train = True
            base_env.HRL = True
            base_env.agent_type = AGENT_TYPE
            base_env.multiple_gaussians = MULTIPLE_GAUSSIANS # testing 2 instead of 1 source at the same time

           
            options = []

            for sp in sub_policies:
                model_path = os.path.join(sp["dir"], sp["model_file"])
                policy = DQN.load(model_path)

                option = Option(
                    policy,
                    sp["termination_fn"],
                    sp["name"],
                    obs_keys=sp["obs_keys"]
                )
                options.append(option)

            # policy1 = DQN.load("../trained-agents/space_+coverage/DQN_scratch.zip")
            # policy2 = DQN.load("../trained-agents/plume_+coverage/DQN_scratch.zip")
            # policy3 = DQN.load("../trained-agents/border_+coverage/DQN_scratch.zip")

            # # note that the termination function dont act: return == FALSE
            # option1 = Option(policy1, termination_fn1, "spatial exploration", obs_keys=["obs_state","visited_maps","space_coverage"])
            # option2 = Option(policy2, termination_fn2, "gas plume exploration", obs_keys=["obs_state","visited_maps", "GT_c_over_threshold_maps", "local_GT","plume_coverage"])
            # option3 = Option(policy3, termination_fn2, "gas border exploration", obs_keys=["obs_state","visited_maps", "GT_c_around_threshold_maps", "local_GT_around", "border_coverage"])

            if GP:
                gp_env = GPWrapper(
                    base_env,
                    kernels_config=kernel_config,
                    gp_visualise=False,
                    HRL=True,
                    SUB_AGENT_TRAIN_ON_GP=False, # we do not train the sugagents rn
                )
            else:
                gp_env = base_env

            env = HierarchicalEnv(
                base_env=gp_env,
                options = options,# options=[option1, option2, option3],
                gamma=0.99,MAX_OPT_LENGTH=MAX_OPT_LENGTH,ACCURACY_GOALS=ACCURACY_GOALS,GP=GP
            )
           
            env = DynamicEpisodeLengthWrapper(env, schedule_fn=episode_length_schedule)

            return env
        return _init
    
    # Create one temp env to extract shape safely
    temp_env = make_env()()
    #obs_space = temp_env.env.base_env.observation_space["visited_maps"].shape # to access  the base env, we need to go first within the dynamic episode wrapper .env and then into the base env with .base_env
    keys_you_want_to_keep = temp_env.env.base_env.observation_space.keys
    N_STATES = temp_env.env.base_env.unwrapped.max_return
    
    #PATCH_RADIUS = temp_env.env.base_env.patch_radius
    temp_env.close()
    # Create and wrap the environment
   
    env_fns = [make_env() for _ in range(NUM_ENVS)]
    env = SubprocVecEnv(env_fns)

    
    policy_kwargs = dict(
	features_extractor_class=AgentHRLExplore           #(PATCH_RADIUS) include it for generalisation
)
    
    load_existing = False  # switch to False to train from scratch

    if load_existing:
        model = DQN.load(
            f"{loading_location}/HRL(ddqn)_scratch",
            env=env,
            tensorboard_log="./tensorboard_logs/"
        )
        model.load_replay_buffer(f"{loading_location}/HRL(ddqn)_replay_buffer.pkl")
    else:
        model = DQN(
            "MultiInputPolicy", # MultiInputPolicy for a dict observation space and MlpPolicy for a flat observation space
            env,
            learning_rate=1e-4,
            buffer_size=100_000, # smaller buffer, HRL acts slower 
            learning_starts=20_000, # learn earlier, as HRL acts slower and we want to have more updates
            batch_size=128,
            gamma=0.99, # =1- 1/33,333 , N=120 because of 3 steps per once
            target_update_interval=10_000, # higher update rate as one step contains more information
            train_freq=1, # as HRL exxecutes many primitive steps
            gradient_steps=1,
            exploration_fraction=0.1, # smaller exploration fraction, as we want to train the high-level policy faster and it has a smaller action space, linearly annealed to final eps over 60% of total steps
            exploration_final_eps=0.05,
            verbose=0,
            policy_kwargs=policy_kwargs, # CNN as feature extractor!!
            tensorboard_log="./tensorboard_logs/",
        )


    
    #training + callback initialisation

    max_steps =  2_000_000
    checkpoint_steps = 15_000_000 // NUM_ENVS # save every n steps, adjusted for number of parallel envs

    stats_callback = EpisodeStatsCallback(max_episode_length=MAX_EPS_LEN,save_freq=checkpoint_steps, NUM_ENVS=NUM_ENVS,address=f"{saving_location}/episode_stats",N_states=N_STATES,GP=GP) # save every n steps, max episode length for _on_step being called only at the end of every episode
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_steps,
        save_path=f"{saving_location}/",
        name_prefix="DQN_checkpoint",
        save_replay_buffer=False # True if desired
    )

    # --- Saving a big DICT with all hyperparameters:---
    build_training_config(model,AGENT_TYPE,keys_you_want_to_keep,None,max_steps,MAX_EPS_LEN,checkpoint_steps,saving_location, kernel_config=kernel_config if GP else None, MAX_OPT_LENGTH=MAX_OPT_LENGTH, MULTIPLE_GAUSSIANS=MULTIPLE_GAUSSIANS,sub_policy_metadata=HRL_sub_policy_data)


    model.learn(total_timesteps=max_steps, callback=[stats_callback, checkpoint_callback], progress_bar=True) 

    # After training
    stats = stats_callback.get_stats()
    stats["waypoints"] = list(range(0, model.num_timesteps+1, checkpoint_steps))
    with open(f"{saving_location}/training_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    # Save the updated model
    model.save(f"{saving_location}/DQN_scratch")
    # save the replay buffer
    #model.save_replay_buffer(f"{saving_location}/DQN_replay_buffer.pkl")
    # Save the updated environment normalization stats
    #env.save(f"{saving_location}/DQN_normalised_scratch.pkl")
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")  # Optional but safe for macOS
    main()