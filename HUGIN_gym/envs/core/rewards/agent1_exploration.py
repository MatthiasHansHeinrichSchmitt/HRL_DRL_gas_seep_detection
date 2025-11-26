import numpy as np

class Reward:
    def __init__(self):
        pass
        
    def get_reward(self, agent_turns,  percentage_visited,diff_new_cells,time_penalty = -0.7,_3D=False, agent_rises=False, VAR_downsampled=None):
        

        reward = 0
        #reward += np.clip(diff_new_cells/4,0,1) # encourage more a lawnmower patern
        reward += diff_new_cells/18
        if agent_turns or agent_rises:
            reward -= 0.3 * (1-percentage_visited)

        if VAR_downsampled:
            reward+= 0.1*(1-4*VAR_downsampled) # max variance is 0.25, with p=0.5
        reward += time_penalty
        return reward
        #print(f"DIFF={diff_new_cells}")
        # if not current_position_IN_visited_locations:
        #     reward+=0.3*diff_new_cells/6+0.3*percentage_unvisited
        # #print(f"DIFF={diff_new_cells}")
            
        # if not fill_position_IN_visited_locations:
        #     reward += 0.3*diff_new_cells/6+0.3*(percentage_unvisited)
        
        # reward += time_penalty

        # if distance_toclosest_unvisited<=1+1e-3:
        #     reward += 0.2
        # else:
        #     reward += 0.2/distance_toclosest_unvisited
        
        # no velocity penalties as the HUGIN has to constantly move 2m/s forward

    
        #[reward is between -1 and 1]
        
