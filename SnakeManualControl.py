from SnakeEnv.SnakeEnv import SnakeEnvironment
import pygame
import sys

SW = 20
SH = 20

if __name__ == "__main__":
    render = True
    env = SnakeEnvironment(screenWidth=SW, screenHeight=SH, render=render)
    for time in range(500):
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            action = 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 1
                if event.key == pygame.K_RIGHT:
                    action = 2

        next_state, reward, done, _ = env.step(action)
        if done:
            break
