import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

import HUGIN_gym  # This import will automatically register the environment


from HUGIN_gym.agents.options import Option
from HUGIN_gym.envs.HRL_env import HierarchicalEnv
from HUGIN_gym.agents.term_fn.termination_fn_agent1 import termination_fn1
from HUGIN_gym.agents.term_fn.termination_fn_agent2 import termination_fn2
import torch.nn as nn
import torch as th
import matplotlib.pyplot as plt

import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF,WhiteKernel, ConstantKernel as C
from HUGIN_gym.envs.core.visualisation.MapVisualiser import MapVisualiser, visualise_state_space,visualise_state_space_hrl 
from HUGIN_gym.envs.core.visualisation.visualise_heatmap import HeatmapVisualiser, CNN
from HUGIN_gym.envs.wrappers.GPWrapper import GPWrapper
from HUGIN_gym.envs.core.visualisation.GP_visualiser import GPVisualiser
from HUGIN_gym.utils.io import load_training_config_from_model
from HUGIN_gym.evaluation.rollout import run_episode_hrl, manual_rollout_hrl,run_episodes_hrl_vec

from stable_baselines3.common.vec_env import SubprocVecEnv

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.network = nn.Sequential(
#             nn.AvgPool2d(kernel_size=3, stride=2),
#             nn.AvgPool2d(kernel_size=3, stride=2),
#             nn.AvgPool2d(kernel_size=3, stride=2),
#             #nn.Flatten()
# )

# =========================================================
# ENV CREATION
# =========================================================

def make_test_env(GP=True, VIS_GP=True, VISUALISE = True,kernel_config=None, MULTIPLE_GAUSSIANS=None, MAX_OPT_LENGTH =None,ACCURACY_GOALS=None):

    base_env = gym.make(
        "HUGIN-v0",
        render_mode="human" if VISUALISE or VIS_GP else None,
        max_episode_steps=4
    )
    base_env = base_env.unwrapped
    base_env.train = True
    base_env.GP_ON = GP
    base_env.random_points = True
    base_env.agent_type ="META"
    base_env.HRL = True # if False we can get the base rewards and overwrite them, but still get the 40/-4 filtering
    base_env.multiple_gaussians = [1,2] if MULTIPLE_GAUSSIANS is None else MULTIPLE_GAUSSIANS # default = [1,1] , it defines the range, e.g. [1,5] would cause from 1 up to 5 gaussian sources within the volume at the same time

     # ---- GP WRAPPER HERE ----
    if GP:
        if kernel_config is None:
            kernel_config = {
            "RBF": {"type": "RBF", "length_scale": 3.5},
            #"Matern": {"type": "Matern", "length_scale": 3.5, "nu": 1.5},}
            }
        else:
            kernel_config=kernel_config
        gp_env = GPWrapper(
            base_env,
            kernels_config=kernel_config,
            gp_visualise=VIS_GP,
            HRL=True,
            SUB_AGENT_TRAIN_ON_GP=False,
        )

        gp_vis = GPVisualiser(
            fig_num=3,
            shift=base_env.unwrapped.offset[0], #careful! assuming dim_len X=Y
            kernel_config=kernel_config
        ) if VIS_GP else None
    else:
        gp_vis = None

    # Load LOW-LEVEL policies
    policy1 = DQN.load("../trained-agents/space_GT_not_clipped_RWD/DQN_scratch.zip")
    policy2 = DQN.load("../trained-agents/plume_GP_clipped_RWD/DQN_scratch.zip")
    policy3 = DQN.load("../trained-agents/border_GP_not_clipped_RWD/DQN_scratch.zip")
 
    # note that the termination function dont act: return == FALSE
    option1 = Option(policy1, termination_fn1, "spatial exploration", obs_keys=["obs_state","visited_maps_downsampled","space_coverage"])
    option2 = Option(policy2, termination_fn2, "gas plume exploration", obs_keys=["obs_state","visited_maps_downsampled", "GT_c_over_threshold_maps_downsampled", "local_GT","plume_coverage","space_coverage"])
    option3 = Option(policy3, termination_fn2, "gas border exploration", obs_keys=["obs_state","visited_maps_downsampled", "GT_c_around_threshold_maps_downsampled", "local_GT_around", "border_coverage","space_coverage"])

    gp_env = gp_env if GP else base_env
    env = HierarchicalEnv(
        base_env=gp_env,
        options=[option1, option2, option3],
        gamma=0.99, MAX_OPT_LENGTH=MAX_OPT_LENGTH,
        ACCURACY_GOALS=ACCURACY_GOALS,GP=GP
    )

    return env,gp_vis

# =========================================================
# TEST AGENT
# =========================================================
model_path_global = "../trained-agents/HRL_GP_base_agents_GP_clipped_1or2G/DQN_scratch" #2_sources

def test_agent():

    VISUALISE =False
    GP = True
    VIS_GP = False
    VIS_STATE_SPACE = False
    EPISODES = 1000
    
    # Load config from the same folder
    config = load_training_config_from_model(model_path_global)
    kernel_config = config.get("HRL_HYPERPARAMETERS", {}).get("kernel_config", None)
    MULTIPLE_GAUSSIANS = [1,2]#config.get("HRL_HYPERPARAMETERS", {}).get("MULTIPLE_GAUSSIANS", None)
    print(MULTIPLE_GAUSSIANS)
    MAX_OPT_LENGTH = 3#config.get("HRL_HYPERPARAMETERS", {}).get("MAX_OPT_LENGTH", None)
    ACCURACY_GOALS = [0.7,0.9,0.9]
    env, gp_vis = make_test_env(GP=GP, VIS_GP=VIS_GP, VISUALISE=VISUALISE,kernel_config=kernel_config,MULTIPLE_GAUSSIANS = MULTIPLE_GAUSSIANS,MAX_OPT_LENGTH=MAX_OPT_LENGTH,ACCURACY_GOALS=ACCURACY_GOALS) # add here a kernel_config from the config file !

    # Load HIGH-LEVEL model (the one you trained in train.py)
    # CNN reconstruction for downsampling. NOTE: if you did not train an agent yet, you can simply use the CNN defined
    try:
        model = DQN.load(model_path_global)
        cnn_full = model.policy.q_net.features_extractor.cnn
        # Remove Flatten layer
        cnn = nn.Sequential(*list(cnn_full.children())[:-1])
        print("Loaded CNN from trained DQN model")
    except Exception as e:
        print(f"Failed to load model ({e}), using default CNN")
        cnn=CNN(layers = 3) # the original design was a resulting 4x4 downsampled state space
        


    MAX_EPS_LEN = config["ENVIRONMENT"]["max_episode_length"] # e.g. 90

    env.visualise_env = VISUALISE # allowing the simulator to update within an option

    if VIS_STATE_SPACE:
        obs,_ = env.reset()
        vis_state_space = visualise_state_space(VIS_STATE_SPACE,obs,cnn,MapVisualiser,HeatmapVisualiser,env.base_env) 
    else:
        vis_state_space=[None,None,None,None]

    percentage_verify =[]
    average_turns = []
    average_steps = []
    average_space_coverage = []
    average_plume_coverage =[]
    average_border_coverage =[]
    average_options_chosen =[]
    for ep in range(EPISODES):

        #print(f"\n=== Episode {ep + 1} ===")
        result = run_episode_hrl(env,model,MAX_EPS_LEN,vis_state_space,gp_vis,GP=GP,VISUALISE=VISUALISE)
        average_space_coverage.append(result["space_coverage"])
        average_border_coverage.append(result["border_coverage"])
        average_plume_coverage.append(result["plume_coverage"])
        percentage_verify.append(result["terminated"]) # if the agent successfully terminated
        average_turns.append(result["episode_turns"])
        average_steps.append(result["base_env_steps"])
        average_options_chosen.append(result["options_chosen"])

    print(f"Options_CHOSEN = {np.mean(np.array(average_options_chosen)):.2f} +- {np.std(np.array(average_options_chosen)):.2f}")
    print(f"STEPS = {np.mean(np.array(average_steps)):.2f} +- {np.std(np.array(average_steps)):.2f}")
    print(f"TURNS= {np.mean(np.array(average_turns)):.2f} +- {np.std(np.array(average_turns)):.2f}")
    print(f"SPACE = {np.mean(np.array(average_space_coverage)):.2f} +- {np.std(np.array(average_space_coverage)):.2f}")
    print(f"PLUME = {np.mean(np.array(average_plume_coverage)):.2f} +- {np.std(np.array(average_plume_coverage)):.2f}")
    print(f"BORDER = {np.mean(np.array(average_border_coverage)):.2f} +- {np.std(np.array(average_border_coverage)):.2f}")
    print(f"SUCCESS_RATE = {np.mean(np.array(percentage_verify)):.2f} +- {np.std(np.array(percentage_verify)):.2f}")
    env.close()

    


def validation():

    VISUALISE = False
    GP = True
    VIS_GP = False
    EPISODES = 1000
    
    # Load config from the same folder
    config = load_training_config_from_model(model_path_global)
    kernel_config = config.get("HRL_HYPERPARAMETERS", {}).get("kernel_config", None)
    MULTIPLE_GAUSSIANS = config.get("HRL_HYPERPARAMETERS", {}).get("MULTIPLE_GAUSSIANS", None)
    MAX_OPT_LENGTH = 1#config.get("HRL_HYPERPARAMETERS", {}).get("MAX_OPT_LENGTH", None)
   
    # Load HIGH-LEVEL model (the one you trained in train.py)
    # CNN reconstruction for downsampling. NOTE: if you did not train an agent yet, you can simply use the CNN defined
    try:
        model = DQN.load(model_path_global)
        cnn_full = model.policy.q_net.features_extractor.cnn
        # Remove Flatten layer
        cnn = nn.Sequential(*list(cnn_full.children())[:-1])
        print("Loaded CNN from trained DQN model")
    except Exception as e:
        print(f"Failed to load model ({e}), using default CNN")
        cnn=CNN(layers = 3) # the original design was a resulting 4x4 downsampled state space
        


    MAX_EPS_LEN = config["ENVIRONMENT"]["max_episode_length"] # e.g. 90

    def make_env_fn():
        def _init():
            env, _ = make_test_env(
                GP=GP,
                VIS_GP=VIS_GP,
                VISUALISE=VISUALISE,
                kernel_config=kernel_config,
                MULTIPLE_GAUSSIANS=MULTIPLE_GAUSSIANS,
                MAX_OPT_LENGTH=MAX_OPT_LENGTH
            )
            return env
        return _init

    env = SubprocVecEnv([make_env_fn() for _ in range(12)])

    model = DQN.load(model_path_global)

    results = run_episodes_hrl_vec(
        env,
        model,
        total_episodes=EPISODES,
        max_eps_len=MAX_EPS_LEN
    )

    average_space_coverage=results["space_coverage"]
    average_border_coverage=results["border_coverage"]
    average_plume_coverage=results["plume_coverage"]
    percentage_verify=results["terminated"]# if the agent successfully terminated
    average_turns=results["episode_turns"]
    average_steps=results["steps"]

    print(f"SUCCESS_RATE = {np.mean(np.array(percentage_verify)):.2f} +- {np.std(np.array(percentage_verify)):.2f}")
    print(f"TURNS= {np.mean(np.array(average_turns)):.2f} +- {np.std(np.array(average_turns)):.2f}")
    print(f"STEPS = {np.mean(np.array(average_steps)):.2f} +- {np.std(np.array(average_steps)):.2f}")
    print(f"SPACE = {np.mean(np.array(average_space_coverage)):.2f} +- {np.std(np.array(average_space_coverage)):.2f}")
    print(f"PLUME = {np.mean(np.array(average_plume_coverage)):.2f} +- {np.std(np.array(average_plume_coverage)):.2f}")
    print(f"BORDER = {np.mean(np.array(average_border_coverage)):.2f} +- {np.std(np.array(average_border_coverage)):.2f}")
    env.close()

def manual_control():
    """
    Test the environment with manual controls for debugging
    Keys:
    - a: spacial exploration agent (SPACE)
    - s: plume exploration agent (PLUME)
    - d border exploration agent (BORDER)
    """ 
    VISUALISE = True            # Meshcat / real-world simulator
    VIS_STATE_SPACE = False   # State-Space representation
    GP=False             # GP on/off
    VIS_GP = False             # GP visualisation
    EPISODES = 40               
    MAX_EPS_LEN = 120           # maximal episode length
    MULTIPLE_GAUSSIANS = [1,2]
    MAX_OPT_LENGTH = 3
    ACCURACY_GOALS = [0.7,0.8,0.8]
    #config = load_training_config_from_model(model_path_global)
    #kernel_config = config.get("HRL_HYPERPARAMETERS", {}).get("kernel_config", None)
    kernel_config = None
    if kernel_config is None:
        kernel_config= { #"Matern": {"type": "Matern", "length_scale": 3.5, "nu": 1.5},}# REPLACE with a config term?!
        "RBF": {"type": "RBF", "length_scale": 3.5},
        }
    
    env, gp_vis = make_test_env(GP=GP, VIS_GP=VIS_GP, VISUALISE=VISUALISE,kernel_config=kernel_config,MULTIPLE_GAUSSIANS=MULTIPLE_GAUSSIANS,MAX_OPT_LENGTH=MAX_OPT_LENGTH,ACCURACY_GOALS=ACCURACY_GOALS) 
  
    #env.unwrapped.train=True

    # CNN reconstruction for downsampling. NOTE: if you did not train an agent yet, you can simply use the CNN defined
    try:
        model = DQN.load(model_path_global)
        cnn_full = model.policy.q_net.features_extractor.cnn
        # Remove Flatten layer
        cnn = nn.Sequential(*list(cnn_full.children())[:-1])
        print("Loaded CNN from trained DQN model")
    except Exception as e:
        print(f"Failed to load model ({e}), using default CNN")
        cnn=CNN(layers = 3) # the original design was a resulting 4x4 downsampled state space
        

    
    
    


    obs, _ = env.reset()
    env.base_env.unwrapped.render()
    vis_state_space = visualise_state_space(VIS_STATE_SPACE,obs,cnn,MapVisualiser,HeatmapVisualiser,env) # initialising the state space visualisation

    for _ in range(EPISODES):
        obs, reward, done , info = manual_rollout_hrl(env,MAX_EPS_LEN,VISUALISE,vis_state_space,gp_vis,GP)

    env.close()


if __name__ == "__main__":
    # Choose whether to run trained agent or manual control
    mode = input(
        "Enter mode (1 for trained agent, 2 manual control and 3 for validation mode): "
    )

    if mode == "1":
        test_agent()
    elif mode == "2":
        manual_control()
    elif mode == "3":
        validation()
    else:
        print("Invalid mode selected")
