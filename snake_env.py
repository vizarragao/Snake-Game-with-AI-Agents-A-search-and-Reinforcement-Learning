# snake_env.py
import pygame
import random
import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

CELL_SIZE = 25
FPS = 60

class SnakeEnv:
    def __init__(self, width=20, height=20, render_mode=True):
        self.width = width
        self.height = height
        self.render_mode = render_mode

        if render_mode:
            pygame.init()
            self.display = pygame.display.set_mode(
                (width * CELL_SIZE, height * CELL_SIZE)
            )
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.direction = RIGHT
        mid_x, mid_y = self.width // 2, self.height // 2
        self.snake = [(mid_x, mid_y), (mid_x - 1, mid_y)]
        self._place_food()
        self.done = False
        self.score = 0
        return self._get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def step(self, action):
        # 0 = straight, 1 = left, 2 = right
        self._update_direction(action)

        old_head = self.snake[0]
        old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])

        new_head = self._next_head_pos()

        reward = 0.0

        # -------------------------
        # Death check
        # -------------------------
        if self._is_collision(new_head):
            self.done = True
            reward = -30      # strong penalty
            return self.get_state(), reward, True, {}

        # -------------------------
        # Move snake
        # -------------------------
        self.snake.insert(0, new_head)

        # -------------------------
        # Food eaten
        # -------------------------
        if new_head == self.food:
            reward = 20       # strong reward
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # -------------------------
        # Distance-based shaping
        # -------------------------
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        if new_dist < old_dist:
            reward += 0.2     # moved closer
        else:
            reward -= 0.2     # moved away

        # -------------------------
        # Survival reward
        # -------------------------
        reward += 0.01

        return self.get_state(), reward, False, {}


    def _update_direction(self, action):
        if action == 1:  # left
            self.direction = (self.direction - 1) % 4
        elif action == 2:  # right
            self.direction = (self.direction + 1) % 4

    def _next_head_pos(self):
        x, y = self.snake[0]
        if self.direction == UP:    y -= 1
        if self.direction == DOWN:  y += 1
        if self.direction == LEFT:  x -= 1
        if self.direction == RIGHT: x += 1
        return (x, y)

    def _is_collision(self, pos):
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if pos in self.snake:
            return True
        return False

    def _danger_ahead_left_right(self):
        dangers = []
        for rel in [0, 1, 2]:
            backup = self.direction
            self._update_direction(rel)
            nxt = self._next_head_pos()
            dangers.append(1.0 if self._is_collision(nxt) else 0.0)
            self.direction = backup
        return dangers

    def _direction_one_hot(self):
        arr = [0, 0, 0, 0]
        arr[self.direction] = 1
        return arr

    def _food_direction(self):
        hx, hy = self.snake[0]
        fx, fy = self.food
        return [
            1.0 if fy < hy else 0.0,  # up
            1.0 if fx > hx else 0.0,  # right
            1.0 if fy > hy else 0.0,  # down
            1.0 if fx < hx else 0.0,  # left
        ]

    def _get_state(self):
        return np.array(
            self._danger_ahead_left_right()
            + self._direction_one_hot()
            + self._food_direction(),
            dtype=np.float32,
        )
    
    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # Danger straight
        danger_straight = int(self._is_collision(self._next_head_pos()))

        # Danger left
        self.direction = (self.direction - 1) % 4
        danger_left = int(self._is_collision(self._next_head_pos()))
        self.direction = (self.direction + 1) % 4

        # Danger right
        self.direction = (self.direction + 1) % 4
        danger_right = int(self._is_collision(self._next_head_pos()))
        self.direction = (self.direction - 1) % 4

        # Food direction
        food_dx = np.sign(food_x - head_x)
        food_dy = np.sign(food_y - head_y)

        return np.array([
        danger_straight,
        danger_left,
        danger_right,
        int(self.direction == 0),
        int(self.direction == 1),
        int(self.direction == 2),
        int(self.direction == 3),
        food_dx,
        food_dy
        ], dtype=np.float32)


    def render(self):
        if not self.render_mode:
            return

        self.display.fill((0, 0, 0))

        # Draw snake
        for (x, y) in self.snake:
            pygame.draw.rect(
                self.display,
                (0, 255, 0),
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
            )

        # Draw food
        fx, fy = self.food
        pygame.draw.rect(
            self.display,
            (255, 0, 0),
            (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )

        pygame.display.update()
        self.clock.tick(FPS)

    def render_game_over(self):
        if not self.render_mode:
            return

        font = pygame.font.SysFont("Arial", 40)
        text = font.render(f"Game Over! Score: {self.score}", True, (255, 255, 255))
        rect = text.get_rect(center=(
            self.width * CELL_SIZE // 2,
            self.height * CELL_SIZE // 2
        ))

        self.display.fill((0, 0, 0))
        self.display.blit(text, rect)
        pygame.display.update()

