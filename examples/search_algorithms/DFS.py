from typing import List, Tuple, Set
import numpy as np

# Movement primitives (dx, dy, dz, dyaw)
ACTIONS = {
    0: np.array([ 2.0,  0.0,  0.0,    0.0       ], dtype=np.float32),
    1: np.array([ 1.0,  0.0,  1.0,    0.0       ], dtype=np.float32),
    2: np.array([ 1.0,  0.0, -1.0,    0.0       ], dtype=np.float32),
    3: np.array([ 1.0,  1.0,  0.0,  np.pi/2     ], dtype=np.float32),
    4: np.array([ 1.0, -1.0,  0.0, -np.pi/2     ], dtype=np.float32),
}

# Discretized orientations
THETAS = [0, np.pi/2, np.pi, -np.pi/2]

class Grid3DCoverageSolver:
    def __init__(self, x_max: int, y_max: int, z_max: int, start: Tuple[int,int,int,int]):
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.start = start  # (x, y, z, theta)
        self.total_positions = x_max * y_max * z_max  # ignore theta for coverage

    def in_bounds(self, x:int, y:int, z:int) -> bool:
        return 0 <= x < self.x_max and 0 <= y < self.y_max and 0 <= z < self.z_max

    def get_neighbors(self, state: Tuple[int,int,int,float]) -> List[Tuple[int,int,int,float,int]]:
        """
        Returns possible neighbors of the form:
        (nx, ny, nz, ntheta, action_id)
        """
        x, y, z, theta = state
        neighbors = []
        for action_id, (dx, dy, dz, dtheta) in ACTIONS.items():
            nx = x + dx
            ny = y + dy
            nz = z + dz
            ntheta = (theta + dtheta) % (2*np.pi)
            if self.in_bounds(nx, ny, nz):
                neighbors.append((nx, ny, nz, ntheta, action_id))
        return neighbors

    def solve(self) -> List[int]:
        """
        Returns a sequence of actions to cover all (x,y,z) positions at least once.
        Theta is considered for transitions but not for coverage counting.
        """
        visited_positions: Set[Tuple[int,int,int]] = set()
        path_actions: List[int] = []

        def dfs(state: Tuple[int,int,int,float]) -> bool:
            x, y, z, theta = state
            visited_positions.add((x,y,z))

            if len(visited_positions) == self.total_positions:
                return True  # all positions visited

            for nx, ny, nz, ntheta, action_id in self.get_neighbors(state):
                if (nx, ny, nz) not in visited_positions:
                    path_actions.append(action_id)
                    if dfs((nx, ny, nz, ntheta)):
                        return True
                    # backtrack
                    path_actions.pop()
            visited_positions.remove((x,y,z))
            return False

        dfs(self.start)
        return path_actions


# Example Usage
if __name__ == "__main__":
    start_state = (0, 0, 0, 0)  # (x, y, z, theta)
    solver = Grid3DCoverageSolver(x_max=4, y_max=4, z_max=1, start=start_state)
    actions = solver.solve()
    print("Action sequence:", actions)
    print("Number of actions:", len(actions))