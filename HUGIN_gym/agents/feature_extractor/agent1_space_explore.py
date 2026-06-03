
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


import numpy as np


    
class AgentSpaceExplore(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)

        state_dim = 7+7*49+1+4*16#+1 #32#observation_space["obs_state"].shape #25 local patch,3 position, 4 angle
        #map_shape = (1,41,41) # shape(Z,X,Y) , as this comes from the observation_space definition and not the batch

        # self.cnn = nn.Sequential(
        # #     #nn.AdaptiveAvgPool2d((5,5)),
        # #     nn.AvgPool2d(kernel_size=3, stride=2),
        # #     nn.AvgPool2d(kernel_size=3, stride=2),
        # #     nn.AvgPool2d(kernel_size=3, stride=2),

        #     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2), #map_shape[0]=Z !!
        #     nn.ReLU(),
        
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2),
        #     nn.ReLU(),

        #     #41x41 needs more reduction in space
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )

        # # compute CNN output size
        # with th.no_grad():
        #     dummy = th.zeros(1, *map_shape) # simulating (batch_size=1, Z,X,Y)
        #     cnn_out = self.cnn(dummy).shape[1]
        #     #cnn_out = 0 #!
            
        self.fc = nn.Sequential(
            nn.Linear( state_dim, 128),#+cnn_out
            nn.ReLU(),
            
        )
        #print(state_dim)
        self._features_dim = 128 # real definition of the feature dimension

    def forward(self, obs):
        state0 = obs["space_coverage"]
        state1 = obs["obs_state"][:, 0:7] # x,y,z, [_ _ _ _] = 7 params (pos., orientation)
        
        state2 = obs["obs_state"][:, 8:351]#57]# # 89#local patch of visited cells
        #print(np.reshape(obs["visited_maps_downsampled"],(41,41,1)))
        visited = obs["visited_maps_downsampled"]#16 # shape = (batch_size,Z,X,Y) and picking the Z=0 layer
        #VAR = obs["VAR_visited_downsampled"]
           # (B, 1681)
      
        return self.fc(th.cat([state0,state1, state2, visited], dim=1))
        #return self.fc(th.cat([state0,state1, state2, visited,VAR], dim=1)) # coverage and state information compression by fc layer
    
     
    