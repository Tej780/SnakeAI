import pygame
import sys
from random import randint
import numpy as np

fps = 15
RED = [255, 0, 0]
GREEN = [0, 255, 0]
DARK_GREEN = [0,128,0]

class SnakeEnvironment:

    def __init__(self, screenWidth=10, screenHeight=10, render=False):
        self.screenSize = [screenWidth, screenHeight]
        self.segments = []
        for i in range(5):
            self.segments.append([screenWidth / 2, (screenHeight / 2) + 1])


        self.apple = self.randomLocation()
        self.died = False
        self.actions = ['FORWARD', 'LEFT', 'RIGHT']
        self.directions = [[0, -1], [1, 0], [0, 1], [-1, 0]] #[UP,RIGHT,DOWN,LEFT]
        self.direction = 0
        obstacles = self.obstacle()
        self.state = [self.segments[0][0], self.segments[0][1], self.apple[0], self.apple[1],
                      self.xDirection(), self.yDirection(),
                      obstacles[0], obstacles[1], obstacles[2],
                      self.distanceToApple(), self.angleToApple()]
        self.totalReward = 0
        self.render = render
        if self.render:
            self.initialiseRender()

    def randomLocation(self):
        return [randint(0, self.screenSize[0]), randint(0, self.screenSize[1])]

    def step(self, actionArg):

        stepReward = 0
        action = self.actions[actionArg]

        distance = self.distanceToApple()

        for i in range(len(self.segments)-1,0,-1):
            self.segments[i] = self.segments[i-1]

        if action == 'FORWARD':
            self.segments[0] = self.front()
        elif action == 'LEFT':
            self.segments[0] = self.left()
            self.direction = (self.direction - 1) % 4
        elif action == 'RIGHT':
            self.segments[0] = self.right()
            self.direction = (self.direction + 1) % 4

        newDistance = self.distanceToApple()

        if newDistance < distance:
            stepReward = 1/(newDistance+0.0000000001)
        for i in range(len(self.segments)-1):
            if self.segments[0] == self.segments[i+1]:
                stepReward = -1
                self.died = True
                break
        if self.isInWall(self.segments[0]):
            stepReward = -1
            self.died = True

        if self.segments[0] == self.apple:
            stepReward = 1
            self.apple = self.randomLocation()
            self.grow()

        obstacles = self.obstacle()
        newState = [self.segments[0][0], self.segments[0][1], self.apple[0], self.apple[1],
                    self.xDirection(), self.yDirection(),
                    obstacles[0],obstacles[1],obstacles[2],
                    self.distanceToApple(), self.angleToApple()]
        self.totalReward += stepReward
        if self.render:
            self.updateDisplay()

        return newState, stepReward, self.died, {}

    def initialiseRender(self):

        pygame.init()
        self.screen = pygame.display.set_mode(((self.screenSize[0]+1) * 10, (self.screenSize[1]+1) * 10))
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
        #self.clock.tick(fps)

    def updateDisplay(self):
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.screen.fill((0, 0, 0))
        player = pygame.Rect(self.segments[0][0] * 10, self.segments[0][1] * 10, 9, 9)
        pygame.draw.rect(self.screen, GREEN, player)
        for i in range(len(self.segments)-1):
            player = pygame.Rect(self.segments[i+1][0] * 10, self.segments[i+1][1] * 10, 9, 9)
            pygame.draw.rect(self.screen, DARK_GREEN, player)
        apple = pygame.Rect(self.apple[0] * 10, self.apple[1] * 10, 10, 10)
        pygame.draw.rect(self.screen, RED, apple)
        pygame.display.update()

        #self.clock.tick(fps)

    def distanceToApple(self):
        return np.sqrt((self.segments[0][0] - self.apple[0]) ** 2
                       + (self.segments[0][1] - self.apple[1]) ** 2)

    def angleToApple(self):
        return np.arctan2(self.apple[0] - self.segments[0][0], self.apple[1] - self.segments[0][1])

    def xDirection(self):
        return self.directions[self.direction][0]

    def yDirection(self):
        return self.directions[self.direction][1]

    def grow(self):
        penultimateSegment = self.segments[-2]
        finalSegment = self.segments[-1]
        directionOfBody = [finalSegment[0]-penultimateSegment[0],finalSegment[1]-penultimateSegment[1]]
        newSegment = [finalSegment[0]-directionOfBody[0],finalSegment[1]-directionOfBody[1]]
        self.segments.append(newSegment)

    def left(self):
        return [self.segments[0][0] + self.directions[(self.direction - 1) % 4][0],
                self.segments[0][1] + self.directions[(self.direction - 1) % 4][1]]

    def right(self):
        return [self.segments[0][0] + self.directions[(self.direction + 1) % 4][0],
                 self.segments[0][1] + self.directions[(self.direction + 1) % 4][1]]

    def front(self):
        return [self.segments[0][0] + self.xDirection(),
                 self.segments[0][1] + self.yDirection()]

    def isInWall(self,coords):
        return coords[0] < 0 or coords[0] > self.screenSize[0] \
               or coords[1] < 0 or coords[1] > self.screenSize[1]

    def obstacle(self):
        obstacleOnLeft = False
        obstacleOnRight = False
        obstacleInFront = False

        if self.isInWall(self.right()):
            obstacleOnRight = True
        if self.isInWall(self.left()):
            obstacleOnLeft = True
        if self.isInWall(self.front()):
            obstacleInFront = True

        for i in range(len(self.segments)-1):
            if self.right() == self.segments[i+1]:
                obstacleOnRight = True
            if self.left() == self.segments[i + 1]:
                obstacleOnLeft = True
            if self.front() == self.segments[i + 1]:
                obstacleInFront = True

        return [obstacleInFront,obstacleOnRight,obstacleOnLeft]
