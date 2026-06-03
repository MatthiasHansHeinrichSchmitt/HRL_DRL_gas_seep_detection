import numpy as np

# Termination for option 2 (gas plume exploration)
def termination_fn2(obs, next_obs, threshold1 = 0.3):
    should_terminate1 = False#should_terminate1 = ((np.sum(next_obs["c_over_threshold_maps"][0])<=np.sum(obs["c_over_threshold_maps"][0])) )# 1st condition: if new move doesnt lead to new visited state.
    should_terminate2 = False #next_obs["obs_state"][7] < 0.75*threshold1 # 2nd: the concentration exploration only works above a ceratin concentration
    should_terminate = should_terminate1 or should_terminate2
    return should_terminate
