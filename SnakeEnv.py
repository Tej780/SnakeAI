import sys
from random import randint

import pygame

fps = 50
RED = [255, 0, 0]
GREEN = [0, 255, 0]
DARK_GREEN = [0, 128, 0]


class SnakeEnvironment:

    def __init__(self, screenSize=10, render=False):
        self.screenSize = screenSize
        self.segments = []
        for i in range(5):
            self.segments.append([screenSize / 2, (screenSize / 2) + 1])

        self.apple = self.randomLocation()
        self.died = False
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', None]
        self.directions = [[0, -1], [1, 0], [0, 1], [-1, 0]]  # [UP,RIGHT,DOWN,LEFT]
        self.direction = [0, -1]

        self.state = self.get_state()
        self.totalReward = 0
        self.render = render
        if self.render:
            self.initialiseRender()

    def randomLocation(self):
        return [randint(0, self.screenSize), randint(0, self.screenSize)]

    def step(self, actionArg):

        stepReward = 0
        apples_collected = 0
        distance_to_apple = self.distance_to_apple()
        action = self.actions[actionArg]

        for i in range(len(self.segments) - 1, 0, -1):
            self.segments[i] = self.segments[i - 1]

        if action == 'DOWN' and self.direction != self.directions[0]:

            self.segments[0] = [self.segments[0][0] + 0,
                                self.segments[0][1] + 1]
            self.direction = self.directions[2]

        elif action == 'LEFT' and self.direction != self.directions[1]:

            self.segments[0] = [self.segments[0][0] + -1,
                                self.segments[0][1] + 0]
            self.direction = self.directions[3]
        elif action == 'UP' and self.direction != self.directions[2]:

            self.segments[0] = [self.segments[0][0] + 0,
                                self.segments[0][1] + -1]
            self.direction = self.directions[0]
        elif action == 'RIGHT' and self.direction != self.directions[3]:

            self.segments[0] = [self.segments[0][0] + 1,
                                self.segments[0][1] + 0]
            self.direction = self.directions[1]
        else:

            self.segments[0] = [self.segments[0][0] + self.direction[0],
                                self.segments[0][1] + self.direction[1]]

        new_distance = self.distance_to_apple()

        if new_distance < distance_to_apple:
            stepReward += 1
        for i in range(len(self.segments) - 1):
            if self.segments[0] == self.segments[i + 1]:
                stepReward += -10
                self.died = True
                break
        if self.isInWall(self.segments[0]):
            stepReward += -10
            self.died = True

        if self.segments[0] == self.apple:
            stepReward += 10
            self.apple = self.randomLocation()
            self.grow()
            apples_collected += 1

        new_state = self.get_state()
        self.totalReward += apples_collected
        if self.render:
            self.updateDisplay()

        return new_state, stepReward, self.died, apples_collected

    def initialiseRender(self):

        pygame.init()
        self.screen = pygame.display.set_mode(((self.screenSize + 1) * 10, (self.screenSize + 1) * 10))
        pygame.mouse.set_visible(0)
        self.clock = pygame.time.Clock()

        player = pygame.Rect(self.segments[0][0] * 10, self.segments[0][1] * 10, 9, 9)
        pygame.draw.rect(self.screen, GREEN, player)
        for i in range(len(self.segments) - 1):
            player = pygame.Rect(self.segments[i + 1][0] * 10, self.segments[i + 1][1] * 10, 9, 9)
            pygame.draw.rect(self.screen, DARK_GREEN, player)

        apple = pygame.Rect(self.apple[0] * 10, self.apple[1] * 10, 10, 10)
        pygame.draw.rect(self.screen, RED, apple)

        pygame.display.flip()
        self.clock.tick(fps)

    def updateDisplay(self):
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.screen.fill((0, 0, 0))
        player = pygame.Rect(self.segments[0][0] * 10, self.segments[0][1] * 10, 9, 9)
        pygame.draw.rect(self.screen, GREEN, player)
        for i in range(len(self.segments) - 1):
            player = pygame.Rect(self.segments[i + 1][0] * 10, self.segments[i + 1][1] * 10, 9, 9)
            pygame.draw.rect(self.screen, DARK_GREEN, player)
        apple = pygame.Rect(self.apple[0] * 10, self.apple[1] * 10, 10, 10)
        pygame.draw.rect(self.screen, RED, apple)
        pygame.display.update()

        self.clock.tick(fps)

    def grow(self):
        penultimateSegment = self.segments[-2]
        finalSegment = self.segments[-1]
        directionOfBody = [finalSegment[0] - penultimateSegment[0], finalSegment[1] - penultimateSegment[1]]
        newSegment = [finalSegment[0] - directionOfBody[0], finalSegment[1] - directionOfBody[1]]
        self.segments.append(newSegment)

    def isInWall(self, coords):
        return coords[0] < 0 or coords[0] > self.screenSize \
               or coords[1] < 0 or coords[1] > self.screenSize

    def distance_to_apple(self):
        # using taxi cab metric
        return abs(self.segments[0][0] - self.apple[0]) + abs(self.segments[0][1] - self.apple[1])

    def get_state(self):
        state = [
            # Player distance to each wall
            self.segments[0][0], self.segments[0][1],
            self.screenSize - self.segments[0][0], self.screenSize - self.segments[0][1],
            # apple distance to each wall
            self.apple[0], self.apple[1],
            self.screenSize - self.apple[0], self.screenSize - self.apple[1]]

        for i in range(len(state)):
            state[i] = state[i] / self.screenSize
        return state
