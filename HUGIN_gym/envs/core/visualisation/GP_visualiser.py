import matplotlib.pyplot as plt


class GPVisualiser:
    def __init__(self, fig_num,shift, kernel_config = None):
        self.shift = shift
        self.fig_num = fig_num
        # Layout:
        # [true] + [kernels...] + [measurements]
        if kernel_config is not None:
            self.total_plots = 2 + len(kernel_config)
            self.fig = plt.figure(fig_num,figsize=(5 * self.total_plots, 4))
        else:
            self.total_plots = 2 + 2
            plt.figure(fig_num, figsize=(20, 4))
        

    def update(self, true_map, predictions, real_grid):
        """
        predictions: dict -> {kernel_name: grid}
        """
        plt.figure(self.fig_num)
        plt.clf()

        plot_idx = 1

        # ---- TRUE MAP ----
        plt.subplot(1, self.total_plots, plot_idx)
        plt.title("True concentration")
        plt.imshow(
            true_map.T, # because the first index in imshow is Y and not X!!
            cmap="magma",
            vmin=0,
            vmax=1,
            extent=[-self.shift, self.shift, -self.shift, self.shift],
            origin="lower",
        )
        plt.ylabel("X")
        plt.xlabel("Y")
        plt.colorbar()
        plot_idx += 1

        # ---- GP PREDICTIONS (dynamic) ----
        for name, grid in predictions.items():
            plt.subplot(1, self.total_plots, plot_idx)
            plt.title(f"GP: {name}")
            plt.imshow(
                grid,
                cmap="magma",
                extent=[-self.shift, self.shift, -self.shift, self.shift],
                origin="lower",
            )
            plt.xlabel("Y")
            plt.colorbar()
            plot_idx += 1

        # ---- MEASUREMENTS ----
        plt.subplot(1, self.total_plots, plot_idx)
        plt.title("Measurements")
        plt.imshow(
            real_grid,
            cmap="magma",
            extent=[-self.shift, self.shift, -self.shift, self.shift],
            origin="lower",
        )
        plt.xlabel("Y")
        plt.colorbar()

        #plt.tight_layout()
        plt.pause(0.01)