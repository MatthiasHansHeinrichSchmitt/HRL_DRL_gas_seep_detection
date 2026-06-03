import matplotlib.pyplot as plt
import numpy as np

class MapVisualiser:
    """
    Efficient visualization of multiple maps without recreating the figure.
    """
    def __init__(self, obs, env, cols=5):
        # Prepare maps to visualize
        self.env = env
        self.maps_to_plot = [
            ("visited", self.env.unwrapped.state.get("visited_map"), "state:visited_map"),
            (r"$c>c_t$", self.env.unwrapped.state.get("c_over_threshold_map"), "state:c_over_threshold_map"),
            (r"$c\sim c_t$", self.env.unwrapped.state.get("c_around_threshold_map"), "state:c_around_threshold_map"),
            (r"GP:$c>c_t$", getattr(self.env.unwrapped, "GT_c_over_threshold_maps"), "env:GT_c_over_threshold_maps"),
            (r"GP:$c\sim c_t$", getattr(self.env.unwrapped, "GT_c_around_threshold_maps"), "env:GT_c_around_threshold_maps"),

            ("(local):visited", obs.get("obs_state")[-147:-98],"obs_state"),
            (r"(local):$c>c_t$", obs.get("obs_state")[-98:-49],"obs_state"),
            (r"(local):$c\sim c_t$", obs.get("obs_state")[-49:],"obs_state"),
            (r"GP(local):$c>c_t$", obs.get("local_GT"), "obs:local_GT"),
            (r"GP(local):$c\sim c_t$", obs.get("local_GT_around"), "obs:local_GT_around")   
        ]
        self.maps_to_plot = [(name, m, key) for name, m, key in self.maps_to_plot if m is not None]

        self.n_maps = len(self.maps_to_plot)
        self.cols = cols
        self.rows = (self.n_maps + cols - 1) // cols

        # Create figure once
        
        self.fig, self.axes = plt.subplots(self.rows, self.cols, figsize=(3*self.cols,2.5*self.rows),constrained_layout=True)
        self.axes = np.array(self.axes).flatten()
        self.fig.suptitle(f"Step {0}", fontsize=16)
        # Create imshow objects for each map
        self.images = []
        for i, (name, m, key) in enumerate(self.maps_to_plot):
            
            source = key.split(":")[0] if ":" in key else key
            grid = self._to_2d(m, source=source)

            # For local maps, just show as-is
            if "local" in name.lower():
                im = self.axes[i].imshow(
                    grid,
                    cmap='magma',
                    interpolation='nearest',
                    vmin=0, vmax=1
                )
            else:
                # Keep your previous global map plotting with extent if you want
                H, W = grid.shape
                extent = [-(H-1)/2, (H-1)/2, -(W-1)/2, (W-1)/2]
                im = self.axes[i].imshow(
                    grid, cmap='magma', interpolation='nearest',
                    vmin=0, vmax=1, extent=extent, origin='lower'
                )
                self.axes[i].invert_xaxis()

            self.axes[i].set_title(name)
            self.axes[i].set_xlabel("Y")
            self.axes[i].set_ylabel("X")
            self.images.append(im)

        # hide unused axes
        for j in range(i+1, len(self.axes)):
            self.axes[j].axis('off')

        # Add single colorbar for all subplots
        self.cbar = self.fig.colorbar(self.images[0], ax=self.axes[:self.n_maps], orientation='vertical')
        self.cbar.set_label('Value')

        plt.ion()
        #plt.tight_layout()
        plt.show()

    def _to_2d(self, m, source=None):
        m = np.array(m)

        if source == "state":
            # (x, y, z, 1) → (z, x, y)
            #m = np.transpose(m[:, :, :, 0], (2, 0, 1))
            m = m[:,:,:,0]

        # env + obs already (z, x, y)

        if m.ndim == 3:
            return m[:,:,0]

        elif m.ndim == 2:
            return m

        elif m.ndim == 1:
            return m.reshape((7, 7))

        else:
            raise ValueError(f"Unexpected shape: {m.shape}")

    def update(self, obs, step):
        for i, (name, _, key) in enumerate(self.maps_to_plot):
            
            if key.startswith("state:"):
                state_key = key.split("state:")[1]
                grid = self._to_2d(self.env.unwrapped.state[state_key][:,:,:,0])

            elif key.startswith("env:"):
                env_key = key.split("env:")[1]
                grid = self._to_2d(getattr(self.env.unwrapped, env_key))

            elif key == "obs_state":
                if name == r"(local):$c\sim c_t$":
                    grid = self._to_2d(obs[key][-49:])
                elif name == r"(local):$c>c_t$":
                    grid = self._to_2d(obs[key][-98:-49])
                elif name == "(local):visited":
                    grid = self._to_2d(obs[key][-147:-98])

            elif key.startswith("obs:"):
                obs_key = key.split("obs:")[1]
                grid = self._to_2d(obs[obs_key])

            else:
                raise ValueError(f"Unknown key type: {key}")

            self.images[i].set_data(grid)

        self.fig.suptitle(f"Step {step}", fontsize=16)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class CoverageBarVisualiser:
    def __init__(self, obs):
        self.labels = ["Space", "Plume", "Border"]

        def _to_scalar(x):
            try:
                if isinstance(x, np.ndarray):
                    return float(x.squeeze())
                if isinstance(x, (list, tuple)):
                    return float(x[0])
                return float(x)
            except Exception as e:
                print(f"{e}couldnt convert cause doesnt have acces to the variavle")
                return -1.0

        self.values = [
            _to_scalar(obs.get("space_coverage", 0)),
            _to_scalar(obs.get("plume_coverage", 0)),
            _to_scalar(obs.get("border_coverage", 0))
        ]

        # ---- Figure ----
        self.fig, self.ax = plt.subplots(figsize=(6, 2.5))

        y_pos = np.arange(len(self.labels))

        # Background bars (full length = 1)
        self.ax.barh(y_pos, [1]*3, color='lightgray')

        # Foreground bars (actual values)
        self.bars = self.ax.barh(y_pos, self.values, color=['blue', 'orange', 'green'])

        self.ax.set_xlim(0, 1)
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(self.labels)
        self.ax.set_xlabel("Coverage")
        self.ax.set_title("Coverage Progress")

        # Percentage text
        self.texts = []
        for i, val in enumerate(self.values):
            txt = self.ax.text(
                val + 0.02,
                i,
                f"{val*100:.1f}%",
                va='center'
            )
            self.texts.append(txt)

        plt.ion()
        plt.show()

    def update(self, obs):
        def _to_scalar(x):
            try:
                if isinstance(x, np.ndarray):
                    return float(x.squeeze())
                if isinstance(x, (list, tuple)):
                    return float(x[0])
                return float(x)
            except Exception as e:
                print(f"{e}couldnt convert cause doesnt have acces to the variavle")
                return -1.0

        new_values = [
            _to_scalar(obs.get("space_coverage", 0)),
            _to_scalar(obs.get("plume_coverage", 0)),
            _to_scalar(obs.get("border_coverage", 0))
        ]

        for bar, val in zip(self.bars, new_values):
            bar.set_width(val)

        for i, (txt, val) in enumerate(zip(self.texts, new_values)):
            txt.set_x(val + 0.02)
            txt.set_text(f"{val*100:.1f}%")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def visualise_state_space(VIS_STATE_SPACE, obs, cnn, MapVisualiser,HeatmapVisualiser,gp_env, all_heatmaps = False,CoverageBarVisualiser = None):
    if VIS_STATE_SPACE:
        vis_state = MapVisualiser(obs,gp_env) 
        vis_heatmap = HeatmapVisualiser(fig_num=2,fig_size=3)
        if all_heatmaps:
            vis_heatmap2 = HeatmapVisualiser(fig_num=4,fig_size=3)
            vis_heatmap3 = HeatmapVisualiser(fig_num=6,fig_size=3)
        vis_obs = obs  
        vis_cnn = cnn
    else:
        vis_state,vis_heatmap, vis_obs,vis_cnn = None,None,None,None

    return_list = [vis_state,vis_heatmap, vis_obs,vis_cnn]
    if all_heatmaps and VIS_STATE_SPACE:
        return_list.append(vis_heatmap2)
        return_list.append(vis_heatmap3)
    if CoverageBarVisualiser:
        return_list.append(CoverageBarVisualiser(obs))
    return return_list

def visualise_state_space_hrl(VIS_STATE_SPACE, cnn, MapVisualiser,HeatmapVisualiser):
    if VIS_STATE_SPACE:
        vis_state = MapVisualiser 
        vis_heatmap = HeatmapVisualiser(fig_num=2,fig_size=3)
        vis_cnn = cnn
    else:
        vis_state,vis_heatmap,vis_cnn = None,None,None
    return [vis_state,vis_heatmap,vis_cnn]