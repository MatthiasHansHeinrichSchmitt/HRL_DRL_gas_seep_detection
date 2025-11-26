import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np


class HUGINRenderer:

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.vis = meshcat.Visualizer()
        self.vis.open()
        self.trail_markers = []  # For trajectory markers
        self.step_counter = 0    # To track how many steps have passed
        self.visited_voxel_names = set() 

    def create_wire_3d_grid(self, size, depth, step):

    

        half_size = size / 2
        half_depth = depth / 2

        vertices = []

        for x in np.arange(-half_size, half_size + step, step):
            for y in np.arange(-half_size, half_size + step, step):
                vertices.append([x, y, -half_depth])
                vertices.append([x, y, half_depth])

        for x in np.arange(-half_size, half_size + step, step):
            for z in np.arange(-half_depth, half_depth + step, step):
                vertices.append([x, -half_size, z])
                vertices.append([x, half_size, z])

        for y in np.arange(-half_size, half_size + step, step):
            for z in np.arange(-half_depth, half_depth + step, step):
                vertices.append([-half_size, y, z])
                vertices.append([half_size, y, z])

        vertices = np.array(vertices).T

        grid = g.LineSegments(
            g.PointsGeometry(vertices),
            g.LineBasicMaterial(color=0x222222, transparent=True,
    opacity=0.3)
        )

        self.vis["3d_grid"].set_object(grid)
        

    def create_wire_2d_grid(self, size, step):

        vertices = []
        for i in np.arange(-size/2, size/2 + step, step):
            vertices.append([i, -size/2, 0])
            vertices.append([i, size/2, 0])
            vertices.append([-size/2, i, 0])
            vertices.append([size/2, i, 0])

        vertices = np.array(vertices).T

        grid = g.LineSegments(
            g.PointsGeometry(vertices),
            g.LineBasicMaterial(color=0x222222)
        )

        self.vis["custom_grid"].set_object(grid)
        self.vis["custom_grid"].set_transform(
            tf.translation_matrix([0, 0, 0.1])
        )
        
    def render(self, model_path):
        size = 50 # world dimensions max(self.x_map_size,self.y_map_size), raise warning if self.x_map_size neq y map size
        depth = 10 # world depth , self.z_map_size
        step = 5
        if self.render_mode != "human":
            return
        
        #water_volume = g.Box([size*20, size*20, depth*20]) # the big water cube
        #water_volume_material = g.MeshPhongMaterial(
         #   color=0x1A6B9F, opacity=0.5, transparent=True
        #)
        
        #self.vis["water_volume"].set_object(water_volume, water_volume_material)
        
        print("model_path: ", model_path)
        self.vis["vessel"].set_object(
            g.DaeMeshGeometry.from_file(model_path),
            g.MeshLambertMaterial(color=0xF58427, wireframe=False),
        )

        ground = g.Box([size, size, 0.01]) # the gray thing at the bottom
        ground_material = g.MeshPhongMaterial(color=0x808080, side="DoubleSide")
        ground_transform = tf.translation_matrix([0, 0, -depth/2])
        self.vis["ground"].set_object(ground, ground_material)
        self.vis["ground"].set_transform(ground_transform)


        # -- the grid --

        # -remove the default grid by meshcat-
        #self.vis["/Background"].set_property("visible", False) # blue background
        # Dark blue top → lighter blue bottom
        self.vis["/Background"].set_property("top_color", [0.0, 0.4, 0.8])
        self.vis["/Background"].set_property("bottom_color", [0.0, 0.2, 0.3])
        self.vis["/Grid"].set_property("visible", False)
        self.vis["/Axes"].set_property("visible", False)

        # -introduce own grid-
        
        self.create_wire_3d_grid(size,depth,step) #3D
        #self.create_wire_2d_grid(size,step) #2D

        # --initialise visited map--

    
       

    def get_robot_position(self):
        return [self.state["x"], self.state["y"], self.state["z"]]

    def plot_marker(self, position, orientation=None, marker_id=None,size=1,color=0x00FF00):
        name = f"marker_{marker_id}" if marker_id else f"marker_{len(self.trail_markers)}"

        if orientation is not None:
            # Shaft of the arrow
            shaft_length = 0.2*size
            shaft_radius = 0.1*size/2
            shaft_geom = g.Cylinder(height=shaft_length, radius=shaft_radius)
            # Rotate cylinder from Z to X
            shaft_tf = tf.rotation_matrix(np.pi / 2, [0, 1, 0])
            # Translate so base is at origin and points along +X
            

            # Head of the arrow (sphere)
            head_radius = 0.2*size/2
            head_geom = g.Sphere(head_radius)
            head_tf = tf.translation_matrix([0,shaft_length/2, 0])


            # Add shaft and head as subpaths
            self.vis[name]["shaft"].set_object(shaft_geom, g.MeshLambertMaterial(color=color))
            self.vis[name]["shaft"].set_transform(shaft_tf)

            self.vis[name]["head"].set_object(head_geom, g.MeshLambertMaterial(color=color))
            self.vis[name]["head"].set_transform(head_tf)

            # Now apply yaw rotation (around Z) to the whole arrow
            rotation = tf.rotation_matrix(orientation, [0, 0, 1])
        else:
            # No orientation → just a sphere
            self.vis[name].set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=color))
            rotation = np.eye(4)

        # Final placement
        translation = tf.translation_matrix(position)
        final_tf = tf.concatenate_matrices(translation, rotation)
        self.vis[name].set_transform(final_tf)


        self.trail_markers.append(name)


    def plot_target(self, target_position):
        target_sphere = g.Sphere(0.15)
        target_material = g.MeshLambertMaterial(color=0xFF0000)  # Bright red

        self.vis["target"].set_object(target_sphere, target_material)

        transform = tf.translation_matrix(target_position)
        self.vis["target"].set_transform(transform)

    def update_visited_grid(self):
        """
        Create a green voxel only when a cell is visited.
        No pre-existing cubes are needed.
        """
        visited = self.state["visited_map"][:, :, :, 0].copy()  # shape (X,Y,Z,1)

        # Get indices of all currently visited cells
        active_indices = np.argwhere(visited == 1)

        for i, j, k in active_indices:
            name = f"cell_{i}_{j}_{k}"

            # Skip if already created
            if name in self.visited_voxel_names:
                continue
            
            self.visited_voxel_names.add(name)  # mark as created

            # Compute world coordinates
            x = i - visited.shape[0]/2 + 0.5
            y = j - visited.shape[1]/2 + 0.5
            z = k - visited.shape[2]/2 + 0.5  # raise slightly above seabed if needed

            # Create cube (slightly smaller than 1 to avoid z-fighting)
            cube = g.Box([0.9, 0.9, 0.9])
            material = g.MeshBasicMaterial(color=0x00FF00, transparent=True, opacity=0.1)

            self.vis["visited_grid"][name].set_object(cube, material)

            # Set transform
            T = tf.translation_matrix([x, y, z])
            self.vis["visited_grid"][name].set_transform(T)

    def step_sim(self, state):
        self.state = state.copy()  # maybe wrong. check later
        if self.render_mode != "human":
            return

        translation = np.array([self.state["x"], self.state["y"], self.state["z"]])
        rotation_matrix = np.array(
            [
                [np.cos(self.state["theta"]-np.pi/2), -np.sin(self.state["theta"]-np.pi/2), 0],
                [np.sin(self.state["theta"]-np.pi/2), np.cos(self.state["theta"]-np.pi/2), 0],
                [0, 0, 1],
            ]
        )
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation

        self.vis["vessel"].set_transform(transform_matrix)
        # Plot marker every 2 steps
        self.step_counter += 1
        #if self.step_counter % 1 == 0:
        self.plot_marker(position=translation, orientation=(self.state["theta"]-np.pi/2), marker_id=self.step_counter)
        #self.update_visited_grid()

    def plot_concentration_field(self, concentration_fn,
                             gaussian_center,
                             sigma,
                             grid_min=-20, grid_max=20,
                             resolution=0.5):
        """
        Visualizes only points within radius sigma of the Gaussian center.

        Opacity:
        r = sigma   -> opacity = 0.2
        r = 0       -> opacity = 0.7
        """

        cx, cy, cz = gaussian_center

        # Remove old field
        try:
            self.vis["concentration_field"].delete()
        except KeyError:
            # Node does not exist yet, nothing to delete
            pass
    
        xs = np.arange(grid_min, grid_max, resolution)
        ys = np.arange(grid_min, grid_max, resolution)
        zs = np.arange(grid_min, grid_max, resolution)

        idx = 0

        for x in xs:
            for y in ys:
                for z in zs:

                    # Compute radial distance from Gaussian center
                    r = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)

                    # Only visualize inside 1 sigma
                    if r > sigma:
                        continue

                    # Get concentration (optional, but useful for scaling)
                    c = concentration_fn(x, y, z)[0,0,0] # new definition of concentraton_fn

                    # Map opacity: r=sigma → 0.2, r=0 → 0.7
                    opacity = 0.2 + (0.7 - 0.2) * (1 - r/sigma)

                    # Sphere radius proportional to concentration
                    radius = 0.1 * float(c) # we treated the concentration before as .astype(np.float32) to work with the neural net. For the visualisation we need float ...

                    sphere = g.Sphere(radius)

                    material = g.MeshPhongMaterial(
                        color=0xBDFFF9,
                        opacity=opacity,
                        transparent=True
                    )

                    name = f"concentration_field/point_{idx}"
                    idx += 1

                    self.vis[name].set_object(sphere, material)
                    self.vis[name].set_transform(
                        tf.translation_matrix([x, y, z])
                    )

    def plot_concentration_field2(self, x,y,z,concentration_field,
                            above_low_lim_of_threshold_border,
                            resolution=0.5):
        """
        Visualizes only points within radius sigma of the Gaussian center.

        Opacity:
        r = sigma   -> opacity = 0.2
        r = 0       -> opacity = 0.7
        """

        # Remove old field
        try:
            self.vis["concentration_field"].delete()
        except KeyError:
            # Node does not exist yet, nothing to delete
            pass
    
        # --- Mask: only meaningful concentration ---
        mask = concentration_field > above_low_lim_of_threshold_border

        # --- Broadcast grid to full shape ---
        X = x + 0*y + 0*z
        Y = y + 0*x + 0*z
        Z = z + 0*x + 0*y

        # --- Extract filtered points ---
        Xf = X[mask]
        Yf = Y[mask]
        Zf = Z[mask]
        Cf = concentration_field[mask]

        # --- Normalize concentration ---
        c_max = concentration_field.max()
        if c_max > 0:
            Cn = Cf / c_max
        else:
            Cn = Cf  # all zeros edge case

        # --- Opacity mapping ---
        # low concentration → 0.2, high → 0.7
        opacity = 0.2 + (0.7 - 0.2) * Cn

        # --- Radius mapping (optional tuning) ---
        radius = 0.3 * Cn  # or try sqrt/log scaling

        # --- Render ---
        for idx, (xv, yv, zv, r, op) in enumerate(zip(Xf, Yf, Zf, radius, opacity)):

            sphere = g.Sphere(float(r))

            material = g.MeshPhongMaterial(
                color=0xBDFFF9,
                opacity=float(op),
                transparent=True
            )

            name = f"concentration_field/point_{idx}"

            self.vis[name].set_object(sphere, material)
            self.vis[name].set_transform(
                tf.translation_matrix([float(xv), float(yv), float(zv+1.0)]) # raise slightly above seabed = bring in same plane as visited map
            )

    def reset(self):
        self.vis.delete()

        self.trail_markers = []
        self.step_counter = 0
        self.visited_voxel_names = set()

        
        