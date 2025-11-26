import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AgentPlumeExplore(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)

        state_dim = 4+49+49+1+1+3+1#+7+1#+49 #32#observation_space["obs_state"].shape #25 local patch,3 position, 4 angle + plume_coverage
        #map_shape = observation_space["visited_maps"].shape # shape(Z,X,Y) , as this comes from the observation_space definition and not the batch
        #in_channels = map_shape[0] * 2  # visited + c_over
        # self.cnn = nn.Sequential(
            
        #     nn.AvgPool2d(kernel_size=3, stride=2),
        #     nn.AvgPool2d(kernel_size=3, stride=2),
        #     nn.AvgPool2d(kernel_size=3, stride=2),
            

            # nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            # nn.ReLU(),
            # #nn.Dropout(p=0.3),
            # nn.MaxPool2d(2),

            # nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # nn.ReLU(),
            # #nn.Dropout(p=0.3),
            # nn.MaxPool2d(2),

            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # #nn.Dropout(p=0.3),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
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
            #nn.Dropout(p=0.5)
        )
        #print(state_dim)
        self._features_dim = 128 # real definition of the feature dimension

    def forward(self, obs):
        state0 = obs["plume_coverage"] # percentage of coverage
        state0b = obs["space_coverage"] # percentage of coverage
        state1 = obs["obs_state"][:, 0:7] # x,y,z, [_ _ _ _] = 7 params (pos., orientation)
        #state1 = obs["obs_state"][:, 3:7] # x,y,z, [_ _ _ _] = 4 params (orientation)
        
        state1b = obs["obs_state"][:, 8:57] # local visited 7x7 =49
        #state2 = obs["obs_state"][:, 57:106]  # Measured: local c> map
        state3 = obs["local_GT"]
        #visited = obs["visited_maps_downsampled"] # shape = (batch_size,Z,X,Y) and picking the Z=0 layer
        
        #c_over = obs["GT_c_over_threshold_maps_downsampled"]
        distance_from_max = obs["distance_from_max"]
        
        #print(visited.shape,c_over.shape)
        
        #x = th.cat([visited, c_over], dim=1)

        # Pass through CNN
        # x = self.cnn(visited)
        # x2 = self.cnn(c_over)
    
        
        #return self.fc(th.cat([state0,state0b,state1,state1b, state3, visited,c_over], dim=1))
        return self.fc(th.cat([state0,state0b,state1,state1b, state3, distance_from_max], dim=1))
        #return self.fc(th.cat([state0,state1,state1b, state3, distance_from_max], dim=1))

#state2, state3