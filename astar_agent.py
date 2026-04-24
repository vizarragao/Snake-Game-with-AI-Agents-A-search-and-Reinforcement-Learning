# astar_agent.py
import heapq
from snake_env import UP, RIGHT, DOWN, LEFT

class AStarAgent:
    def __init__(self, env):
        self.env = env
        self.path = []

    def act(self, state):
        if not self.path:
            self.path = self._plan_path()
        if not self.path:
            return 0  # straight as fallback

        next_cell = self.path.pop(0)
        return self._cell_to_relative_action(next_cell)

    def _neighbors(self, pos, snake_body):
        x, y = pos
        candidates = [(x, y - 1, UP), (x + 1, y, RIGHT),
                      (x, y + 1, DOWN), (x - 1, y, LEFT)]
        result = []
        for nx, ny, d in candidates:
            if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                if (nx, ny) not in snake_body:
                    result.append((nx, ny))
        return result

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _plan_path(self):
        start = self.env.snake[0]
        goal = self.env.food
        snake_body = set(self.env.snake[1:])  # avoid body, allow head

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current, snake_body):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return []

    def _reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _cell_to_relative_action(self, next_cell):
        head_x, head_y = self.env.snake[0]
        nx, ny = next_cell
        # desired absolute direction
        if ny < head_y:
            desired = UP
        elif ny > head_y:
            desired = DOWN
        elif nx > head_x:
            desired = RIGHT
        else:
            desired = LEFT

        cur = self.env.direction
        if desired == cur:
            return 0  # straight
        # left or right?
        if (cur - desired) % 4 == 1:
            return 1  # left
        else:
            return 2  # right
