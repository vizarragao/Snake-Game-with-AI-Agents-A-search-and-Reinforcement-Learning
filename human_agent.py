# human_agent.py
import pygame
from snake_env import UP, RIGHT, DOWN, LEFT

class HumanAgent:
    def __init__(self, env):
        self.env = env

    def act(self, state):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        keys = pygame.key.get_pressed()

        desired = self.env.direction

        if keys[pygame.K_w]:    desired = UP
        if keys[pygame.K_s]:  desired = DOWN
        if keys[pygame.K_a]:  desired = LEFT
        if keys[pygame.K_d]: desired = RIGHT

        cur = self.env.direction

        if desired == cur:
            return 0  # straight
        if (cur - desired) % 4 == 1:
            return 1  # left
        return 2      # right
