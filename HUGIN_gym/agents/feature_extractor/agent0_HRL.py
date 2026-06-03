import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AgentHRLExplore(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=499)
        state_dim = 7+49+49 +49+3+1#32#observation_space["obs_state"].shape #25 local patch,3 position, 4 angle
        #map_shape = observation_space["visited_maps"].shape # shape(Z,X,Y) , as this comes from the observation_space definition and not the batch

        # self.cnn = nn.Sequential(
            
        #     nn.AvgPool2d(kernel_size=3, stride=2),
        #     nn.AvgPool2d(kernel_size=3, stride=2),
        #     nn.AvgPool2d(kernel_size=3, stride=2),
        #     nn.Flatten()
        # )

        # # compute CNN output size
        # with th.no_grad():
        #     dummy = th.zeros(1, *map_shape) # simulating (batch_size=1, Z,X,Y)
        #     cnn_out = self.cnn(dummy).shape[1]
        #     #cnn_out = 0 #!
            
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        #print(state_dim)
        self._features_dim = 128 # real definition of the feature dimension
    def forward(self, obs):
        state0a = obs["space_coverage"]
        state0b = obs["plume_coverage"]
        state0c = obs["border_coverage"]

        state1 = obs["obs_state"][:, 0:7] # x,y,z, [_ _ _ _] = 7 params (pos., orientation)
        
        state1b = obs["obs_state"][:, 8:57] # local visited 7x7 =49
        state3 = obs["local_GT"]
        state4 = obs["local_GT_around"]

        #visited = obs["visited_maps_downsampled"] # shape = (batch_size,Z,X,Y) and picking the Z=0 layer
        #state3 = obs["local_GT"]
        #c_over = obs["GT_c_over_threshold_maps_downsampled"]
        #c_around = obs["GT_c_around_threshold_maps_downsampled"]
        #VAR = obs["VAR_visited_downsampled"]

        distance_from_max = obs["distance_from_max"]
    
        return self.fc(th.cat([state0a,state0b, state0c, state1, state1b, state3, state4,distance_from_max], dim=1))
        #return self.fc(th.cat([state0a,state0b, state0c, state1, state1b, state3, state4,visited,c_over,c_around], dim=1))
#,state3