import matplotlib.pyplot as plt
import numpy as np

class HeatmapVisualiser:
    def __init__(self, fig_num,fig_size=3):
        """
        Visualises the downsampled grid as a heatmap using Matplotlib.
        """
        plt.ion()
        self.fig = plt.figure(fig_num,figsize=(fig_size, fig_size))
        #self.cbar = None
        self.fig_num=fig_num

    def update(self, grid, step):
        """
        Updates the heatmap with new grid data.
        """
        plt.figure(self.fig_num)
        plt.clf()  # clear the whole figure

        H, W = grid.shape

        extent = [-(H-1)/2, (H-1)/2, -(W-1)/2, (W-1)/2]

        im = plt.imshow(
            grid,
            cmap='magma',
            interpolation='nearest',
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            origin='lower'
        )

        plt.title(f"Heatmap at Step {step}")
        plt.xlabel("Y")
        plt.ylabel("X")

        plt.yticks(np.arange(-(H-1)/2, (H-1)/2 + 1, 1))
        plt.xticks(np.arange(-(W-1)/2, (W-1)/2 + 1, 1))

        plt.gca().invert_xaxis()

        # handle colorbar
        plt.colorbar()

        plt.draw()
        plt.pause(0.1)

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, layers=3):
        super(CNN, self).__init__()

        layer_list = []

        for _ in range(layers):
            layer_list.append(nn.AvgPool2d(kernel_size=3, stride=2))

        # optionally add flatten at the end
        # layer_list.append(nn.Flatten())

        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network(x)
    
class CNN3D(nn.Module):
    def __init__(self, layers=3):
        super(CNN3D, self).__init__()

        layer_list = []

        for _ in range(layers):
            layer_list.append(nn.AvgPool3d(kernel_size=3, stride=2))

        # optionally add flatten at the end
        # layer_list.append(nn.Flatten())

        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network(x)
    