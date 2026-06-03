import numpy as np

# Termination for option 1 (spatial exploration)
def termination_fn1(obs, next_obs, threshold1 = 0.3):
    should_terminate1 = False # 1st condition: if new move doesnt lead to new visited state.-> terminate. NOTE: that visited are 0 and unvisited are 1
    should_terminate2 = False #next_obs["obs_state"][7] > 0.85*threshold1 # stop space explore -> plume explore
    should_terminate = should_terminate1 or should_terminate2
    return should_terminate