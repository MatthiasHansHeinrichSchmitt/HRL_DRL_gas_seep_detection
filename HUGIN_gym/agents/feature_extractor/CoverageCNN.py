import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CoverageCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1) # 1 serves as a placeholder

        state_dim = 32#observation_space["obs_state"].shape #3 position, 4 angle, 1 concentration
        map_shape = observation_space["visited_maps"].shape # shape(Z,X,Y) , as this comes from the observation_space definition and not the batch

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=map_shape[0], out_channels=16, kernel_size=3, stride=2, padding=1), #map_shape[0]=Z !!
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # no strride >1 for small grid as 11x11
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(2),
            # nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # no strride >1 for small grid as 11x11
            # nn.ReLU(),
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # no strride >1 for small grid as 11x11
            # nn.ReLU(),
            #nn.AdaptiveAvgPool2d(3),
            nn.Flatten()
        )

        # compute CNN output size
        with th.no_grad():
            dummy = th.zeros(1, *map_shape) # simulating (batch_size=1, Z,X,Y)
            cnn_out = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out + state_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        #print(state_dim)
        self._features_dim = 33 + 21*21 # real definition of the feature dimension

    def forward(self, obs):
        state = obs["obs_state"][:,0:33] # x,y,z, [_ _ _ _], c = 8 params (pos., orientation, conc.) 
        #visited_patch = obs["obs_state"][:,8:33] #+25 local patch
        visited = obs["visited_maps"][:,0] # shape = (batch_size,Z,X,Y)
        #x = self.cnn(visited)
        
        x = th.cat([state,visited.reshape(visited.shape[0],-1)], dim=1)
        
        return x