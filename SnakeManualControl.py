import sys

import pygame

from SnakeEnv import SnakeEnvironment

SS = 20

if __name__ == "__main__":
    render = True
    env = SnakeEnvironment(screenSize=SS, render=render)
    for time in range(500):
        action = 4
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                if event.key == pygame.K_DOWN:
                    action = 2
                if event.key == pygame.K_LEFT:
                    action = 3
                if event.key == pygame.K_RIGHT:
                    action = 1

        next_state, reward, done, _ = env.step(action)
        if done:
            break
