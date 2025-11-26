import gpytorch
import torch as th

class SpatialGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GPMemory:
    def __init__(self, max_size=5000):
        self.X = []
        self.y = []
        self.max_size = max_size

    def add(self, x, y, z, value):
        if len(self.X) >= self.max_size:
            self.X.pop(0) # release the oldest memory 
            self.y.pop(0)

        self.X.append([x, y, z])
        self.y.append(value)

    def get_tensors(self):
        return th.tensor(self.X).float(), th.tensor(self.y).float()