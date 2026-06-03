import numpy as np



def build_lawnmower_path_from_initial_pose(env_unwrapped):

    max_x = env_unwrapped.map_size_x
    max_y = env_unwrapped.map_size_y
    iz = env_unwrapped.map_size_z // 2

    ix0, iy0, _ = env_unwrapped._pos_to_idx(
        env_unwrapped.state["x"],
        env_unwrapped.state["y"],
        env_unwrapped.state["z"],
    )

    t=theta =env_unwrapped.state["theta"]
  

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

        # +y or -y
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



