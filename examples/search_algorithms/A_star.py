import heapq
import numpy as np
from itertools import product

# ---------- Environment ----------

class Environment3D:
    def __init__(self, x_limits, y_limits, z_limits):
        self.xmin, self.xmax = x_limits
        self.ymin, self.ymax = y_limits
        self.zmin, self.zmax = z_limits

        # Precompute all admissible positions
        self.all_positions = [
            (x, y, z) 
            for x in range(self.xmin, self.xmax + 1)
            for y in range(self.ymin, self.ymax + 1)
            for z in range(self.zmin, self.zmax + 1)
        ]
        # Map positions to indices for bitmask
        self.pos_to_idx = {pos: i for i, pos in enumerate(self.all_positions)}
        self.num_positions = len(self.all_positions)

    def valid(self, x, y, z):
        return (self.xmin <= x <= self.xmax and
                self.ymin <= y <= self.ymax and
                self.zmin <= z <= self.zmax)

# ---------- Heuristic ----------

def heuristic(visited_mask, env):
    # Simple: count of unvisited positions
    unvisited = env.num_positions - bin(visited_mask).count("1")
    return unvisited

# ---------- Yaw handling ----------

YAW_VALUES = [0, np.pi/2, np.pi, 3*np.pi/2]

def normalize_yaw(y):
    return YAW_VALUES[int(round(y / (np.pi/2))) % 4]

# ---------- Actions ----------

# ACTIONS = {
#     0: np.array([ 2,  0,  0,    0], dtype=np.int32),
#     1: np.array([ 1,  0,  1,    0], dtype=np.int32),
#     2: np.array([ 1,  0, -1,    0], dtype=np.int32),
#     3: np.array([ 1,  1,  0,  np.pi/2], dtype=np.float32),
#     4: np.array([ 1, -1,  0, -np.pi/2], dtype=np.float32),
# }
ACTIONS = {
    0: np.array([ 1,  0,  0,    0], dtype=np.int32),
    1: np.array([ 0,  1,  0,    0], dtype=np.int32),
    2: np.array([ 0,  0, 1,    0], dtype=np.int32),
    3: np.array([ -1,  0,  0,    0], dtype=np.int32),
    4: np.array([ 0,  -1,  0,    0], dtype=np.int32),
    5: np.array([ 0,  0, -1,    0], dtype=np.int32),
}

def transition_cost(action_vec, revisited=False):
    dx, dy, dz, dyaw = action_vec
    cost = 1 if dz == 0 else 10000
    if abs(dy)>0:
        cost+=100
    if dyaw != 0:
        cost += 10
    if revisited:
        cost += 1000000000  # heavy penalty for revisiting
    return cost

# ---------- action trajectory --------

def action_trajectory(cx, cy, cz, action):
    """Return all intermediate integer positions that the action passes through."""
    dx, dy, dz, dtheta = action
    return [(cx+dx,cy+dy,cz+dz)]
    # traj = []
    # if dtheta != 0: 
    #     traj.append((cx+int(dx),cy,cz))
    #     traj.append((cx+int(dx),cy+int(dy),cz))
    # elif dx == 2:
    #     traj.append((cx+int(dx/2),cy,cz))
    #     traj.append((cx+int(dx),cy,cz))
    # else:
    #     traj.append((cx+int(dx),cy+int(dy),cz))
    # return traj

# ---------- A* with bitmask ----------

def astar_coverage(env, start, desired_coverage=None, max_expansions=2000000):
    start_pos = (start[0], start[1], start[2])
    if start_pos not in env.pos_to_idx:
        raise ValueError(f"start_pos {start_pos} not in environment grid")

    start_yaw = normalize_yaw(start[3])
    start_mask = 1 << env.pos_to_idx[start_pos]

    if desired_coverage is None:
        desired_coverage = env.num_positions  # full coverage by default


    open_set = []
    heapq.heappush(open_set, (heuristic(start_mask, env), (start[0], start[1], start[2], start_yaw, start_mask)))
    
    came_from = {}
    g_score = {}
    g_score[(start[0], start[1], start[2], start_mask)] = 0

    expansions = 0
    visited_states = 0

    while open_set:
        _, (cx,cy,cz,cyaw,mask) = heapq.heappop(open_set)
        current_key = (cx,cy,cz,mask)
        expansions += 1
        if expansions > max_expansions:
            # safety bailout
            return None


        # check softened goal condition: visited count >= desired_coverage
        if bin(mask).count("1") >= (desired_coverage-1):
            return reconstruct_path(came_from, current_key, env)


        for action_idx, action in ACTIONS.items():
            dx, dy, dz, dyaw = action
            # compute integer endpoint
            nx = int(round(cx + dx))
            ny = int(round(cy + dy))
            nz = int(round(cz + dz))

            # quick out-of-bounds check for endpoint
            if not env.valid(nx, ny, nz):
                continue

            new_yaw = normalize_yaw(cyaw + dyaw)


            # Update bitmask
            # Compute trajectory of all positions the action passes through
            traj = action_trajectory(cx, cy, cz, action)

            # ---- FIX 1: validate full trajectory before setting bits ----
            valid_traj = True
            for pos in traj:
                if not env.valid(*pos):
                    valid_traj = False
                    break
            if not valid_traj:
                continue

            # ---- FIX 2: now apply mask updates safely ----
            new_mask = mask
            revisited = False
            for pos in traj:
                pos_idx = env.pos_to_idx[pos]
                if new_mask & (1 << pos_idx):
                    revisited = True
                new_mask |= (1 << pos_idx)

            # ---- FIX 3: safe neighbor ----
            end_x, end_y, end_z = traj[-1]
            neighbor_node = (end_x, end_y, end_z, new_yaw, new_mask)
            # neighbor key WITHOUT yaw
            neighbor_key = (end_x, end_y, end_z, new_mask)

            tentative_g = g_score[current_key] + transition_cost(action, revisited)
            if tentative_g < g_score.get(neighbor_key, float('inf')):
                came_from[neighbor_key] = (current_key, action_idx)
                g_score[neighbor_key] = tentative_g
                f = tentative_g + heuristic(new_mask, env)
                heapq.heappush(open_set, (f, neighbor_node))



    return None  # No path found

# ---------- Path reconstruction ----------

def reconstruct_path(came_from, current, env):
    path = []
    actions = []

    while current in came_from:
        prev, action_idx = came_from[current]
        path.append(current[:4])  # ignore bitmask in path
        actions.append(action_idx)
        current = prev

    path.append(current[:4])
    path.reverse()
    actions.reverse()
    return path, actions

# ---------- Example ----------

env = Environment3D(
    x_limits=(0, 2),
    y_limits=(0, 2),
    z_limits=(-1, 1)
)
print(env.num_positions)
start = (0, 0, 0, 0)  # x, y, z, yaw
path, actions = astar_coverage(env, start)

print("Path length:", len(path))
print("Actions:", actions)
