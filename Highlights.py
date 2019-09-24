import numpy as np
import pygame

fps = 60


def replay_highlights(episode_list, screenSize):
    pygame.init()
    screen = pygame.display.set_mode(((screenSize + 1) * 10, (screenSize + 1) * 10))
    pygame.display.set_caption('Highlights')
    pygame.mouse.set_visible(0)
    clock = pygame.time.Clock()
    pygame.display.flip()

    for episode in episode_list:

        for frame in episode:
            pygame.surfarray.blit_array(screen, frame)
            pygame.display.flip()
            clock.tick(fps)


if __name__ == "__main__":
    highlights = np.load('Highlights.npy', allow_pickle=True)

    replay_highlights(highlights, 20)
