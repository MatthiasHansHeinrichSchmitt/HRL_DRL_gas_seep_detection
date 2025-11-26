import numpy as np

def build_lawnmower_path_xy_from_pose(env_unwrapped, ix0, iy0, iz, theta0):
    max_x = env_unwrapped.map_size_x
    max_y = env_unwrapped.map_size_y

    t = theta0
    path = []

    # ==========================================================
    # CASE 1:
    # Long transects parallel to X
    # ==========================================================
    long_in_x = (
        np.isclose(t, 0.0, atol=1e-4)
        or np.isclose(t, np.pi, atol=1e-4)
    )

    if long_in_x:

        # +x or -x
        sweep_dir = +1 if np.isclose(t, 0.0, atol=1e-4) else -1

        dist_down = iy0
        dist_up = (max_y - 1) - iy0

        rows_down = dist_down // 2
        rows_up = dist_up // 2

        ix = ix0
        
        # ======================================================
        # FAR SIDE = UP
        # ======================================================
        if dist_up >= dist_down:
            
            for _ in range(rows_up):

                while (ix < (max_x - 3) and sweep_dir == 1) or (
                    ix > 2 and sweep_dir == -1):
                    path.append(0)
                    ix += 2 if sweep_dir == 1 else -2

                if sweep_dir == 1:
                    path.append(3)
                    path.append(3)
                else:
                    path.append(4)
                    path.append(4)

                sweep_dir *= -1

            while (ix < (max_x - 3) and sweep_dir == 1) or (
                    ix > 2 and sweep_dir == -1):
                path.append(0)
                ix += 2 if sweep_dir == 1 else -2

            # transition across center
            if sweep_dir == 1:
                path.append(4)
            else:
                path.append(3)
              
            for _ in range(rows_up - 1):
                path.append(0)

            if sweep_dir == 1:
                path.append(4)
            else:
                path.append(3)
                
            sweep_dir *= -1

            # cover lower side
            for _ in range(rows_down):

                while (ix < max_x - 3 and sweep_dir == 1) or (
                    ix > 2 and sweep_dir == -1
                ):
                    path.append(0)
                    ix += 2 if sweep_dir == 1 else -2

                if sweep_dir == 1:
                    path.append(4)
                    path.append(4)
                else:
                    path.append(3)
                    path.append(3)

                sweep_dir *= -1
            
            while (ix < (max_x - 3) and sweep_dir == 1) or (
                    ix > 2 and sweep_dir == -1):
                path.append(0)
                ix += 2 if sweep_dir == 1 else -2

        # ======================================================
        # FAR SIDE = DOWN
        # ======================================================
        else:

            for _ in range(rows_down):

                while (ix < (max_x - 3) and sweep_dir == 1) or (
                    ix > 2 and sweep_dir == -1):
                    path.append(0)
                    ix += 2 if sweep_dir == 1 else -2

                if sweep_dir == 1:
                    path.append(4)
                    path.append(4)
                else:
                    path.append(3)
                    path.append(3)

                sweep_dir *= -1

            while (ix < (max_x - 3) and sweep_dir == 1) or (
                    ix > 2 and sweep_dir == -1):
                path.append(0)
                ix += 2 if sweep_dir == 1 else -2

            # transition across center
            if sweep_dir == 1:
                path.append(3)
            else:
                path.append(4)
              
            for _ in range(rows_down - 1):
                path.append(0)

            if sweep_dir == 1:
                path.append(3)
            else:
                path.append(4)
                
            sweep_dir *= -1

            # cover lower side
            for _ in range(rows_up):

                while (ix < max_x - 3 and sweep_dir == 1) or (
                    ix > 2 and sweep_dir == -1
                ):
                    path.append(0)
                    ix += 2 if sweep_dir == 1 else -2

                if sweep_dir == 1:
                    path.append(3)
                    path.append(3)
                else:
                    path.append(4)
                    path.append(4)

                sweep_dir *= -1
            
            while (ix < (max_x - 3) and sweep_dir == 1) or (
                    ix > 2 and sweep_dir == -1):
                path.append(0)
                ix += 2 if sweep_dir == 1 else -2

    # ==========================================================
    # CASE 2:
    # Long transects parallel to Y
    # ==========================================================
    else:

        sweep_dir = (
            +1
            if np.isclose(t, 0.5 * np.pi, atol=1e-4)
            else -1
        )

        dist_left = ix0
        dist_right = (max_x - 1) - ix0

        cols_left = dist_left // 2
        cols_right = dist_right // 2

        iy = iy0

        # ======================================================
        # FAR SIDE = RIGHT
        # ======================================================
        if dist_right >= dist_left:

            for _ in range(cols_right):

                while (iy < max_y - 3 and sweep_dir == 1) or (
                    iy > 2 and sweep_dir == -1
                ):
                    path.append(0)
                    iy += 2 if sweep_dir == 1 else -2

                # move right
                if sweep_dir == 1:
                    path.append(4)
                    path.append(4)
                else:
                    path.append(3)
                    path.append(3)

                sweep_dir *= -1

            while (iy < max_y - 3 and sweep_dir == 1) or (
                    iy > 2 and sweep_dir == -1
                ):
                path.append(0)
                iy += 2 if sweep_dir == 1 else -2

            # transition across center
            if sweep_dir == 1:
                path.append(3)
            else:
                path.append(4)

            for _ in range(cols_right - 1):
                path.append(0)

            if sweep_dir == 1:
                path.append(3)
            else:
                path.append(4)

            sweep_dir *= -1

            # cover left side
            for _ in range(cols_left):

                while (iy < max_y - 3 and sweep_dir == 1) or (
                    iy > 2 and sweep_dir == -1
                ):
                    path.append(0)
                    iy += 2 if sweep_dir == 1 else -2

                if sweep_dir == 1:
                    path.append(3)
                    path.append(3)
                else:
                    path.append(4)
                    path.append(4)

                sweep_dir *= -1

            while (iy < max_y - 3 and sweep_dir == 1) or (
                    iy > 2 and sweep_dir == -1
                ):
                path.append(0)
                iy += 2 if sweep_dir == 1 else -2

        # ======================================================
        # FAR SIDE = LEFT
        # ======================================================
        else:
            
            for _ in range(cols_left):

                while (iy < max_y - 3 and sweep_dir == 1) or (
                    iy > 2 and sweep_dir == -1):
                    path.append(0)
                    iy += 2 if sweep_dir == 1 else -2

                # move left
                if sweep_dir == 1:
                    path.append(3)
                    path.append(3)
                else:
                    path.append(4)
                    path.append(4)

                sweep_dir *= -1

            while (iy < max_y - 3 and sweep_dir == 1) or (
                    iy > 2 and sweep_dir == -1):
                path.append(0)
                iy += 2 if sweep_dir == 1 else -2

            # transition across center
            if sweep_dir == 1:
                path.append(4)
            else:
                path.append(3)

            for _ in range(cols_left - 1):
                path.append(0)

            if sweep_dir == 1:
                path.append(4)
            else:
                path.append(3)

            sweep_dir *= -1

            # cover right side
            for _ in range(cols_right):

                while (iy < max_y - 3 and sweep_dir == 1) or (
                    iy > 2 and sweep_dir == -1
                ):
                    path.append(0)
                    iy += 2 if sweep_dir == 1 else -2

                if sweep_dir == 1:
                    path.append(4)
                    path.append(4)
                else:
                    path.append(3)
                    path.append(3)

                sweep_dir *= -1

            while (iy < max_y - 3 and sweep_dir == 1) or (
                    iy > 2 and sweep_dir == -1
                ):
                path.append(0)
                iy += 2 if sweep_dir == 1 else -2

    return path

import numpy as np

def step_diagonal_in_3d(ix, iy, iz, theta, action):
    """
    Update integer indices (ix, iy, iz) for actions 1 (up) and 2 (down),
    assuming theta is one of {0, pi/2, pi, 3pi/2} up to small numerical error.
    """

    assert action in (1, 2), "step_diagonal_in_3d is only for actions 1 and 2"

    # determine forward direction in the grid
    if np.isclose(theta, 0.0, atol=1e-4):
        # +x
        ix_new = ix + 1
        iy_new = iy
    elif np.isclose(theta, 0.5 * np.pi, atol=1e-4):
        # +y
        ix_new = ix
        iy_new = iy + 1
    elif np.isclose(theta, np.pi, atol=1e-4):
        # -x
        ix_new = ix - 1
        iy_new = iy
    else:
        # we assume this is 3*pi/2 or -pi/2 -> -y
        ix_new = ix
        iy_new = iy - 1

    # z change: action 1 = up, action 2 = down
    if action == 1:
        iz_new = iz + 1
    else:
        iz_new = iz - 1

    return ix_new, iy_new, iz_new
from HUGIN_gym.evaluation.lawnmower_path_generator import build_lawnmower_path_from_initial_pose
def build_lawnmower_path_3d_from_initial_pose(env_unwrapped):
    max_x = env_unwrapped.map_size_x
    max_y = env_unwrapped.map_size_y
    max_z = env_unwrapped.map_size_z

    # current discrete index and heading of the agent
    ix, iy, iz = env_unwrapped._pos_to_idx(
        env_unwrapped.state["x"],
        env_unwrapped.state["y"],
        env_unwrapped.state["z"],
    )
    theta = env_unwrapped.state["theta"]

    path = []

    # --- choose z-layer order ---
    # simplest: start where you are, go up to the top, then go down
    z_layers_up = list(range(iz, max_z))       # iz, iz+1, ..., max_z-1
    z_layers_down = list(range(iz - 1, -1, -1))  # iz-1, ..., 0

    # --- go UPWARDS first ---
    if iz >=max_z //2:
        z_layers_up,z_layers_down = z_layers_down,z_layers_up # make sure longer part is done first 
    for target_iz in z_layers_up:
        # 1) mow the current layer (fixed iz)
        #    we reuse your 2D pattern; it assumes a fixed z index.
        #    Note: it does NOT update ix, iy, theta, so we only add actions.
        slice_actions = build_lawnmower_path_from_initial_pose(env_unwrapped)
        path.extend(slice_actions[0:-1])

        # 2) if we still have a higher layer to go to, step up (action 1)
        if target_iz < max_z - 1:
            # action 1 = forward + up
            path.append(1)
            ix, iy, iz = step_diagonal_in_3d(ix, iy, iz, theta, 1)
            # theta does not change for action 1

    # --- then go DOWNWARDS ---
    for target_iz in z_layers_down:
        slice_actions = build_lawnmower_path_from_initial_pose(env_unwrapped)
        path.extend(slice_actions[0:-1])

        if target_iz > 0:
            # action 2 = forward + down
            path.append(2)
            ix, iy, iz = step_diagonal_in_3d(ix, iy, iz, theta, 2)
            # theta does not change for action 2

    return path

