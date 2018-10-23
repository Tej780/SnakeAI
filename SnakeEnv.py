import pygame
import sys
from random import randint
import numpy as np




class SnakeEnvironment():

    def __init__(self, screenWidth = 10, screenHeight = 10,render = False):
        self.screenSize = [screenWidth,screenHeight]
        self.player = [screenWidth/2,screenHeight/2]
        self.apple = self.randomLocation()
        self.died = False
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.state = [self.player[0],self.player[1],self.apple[0],self.apple[1]]
        self.totalReward = 0
        self.render = render
        if self.render == True:
            self.initialiseRender()




    def randomLocation(self):
        return [randint(0, self.screenSize[0]), randint(0, self.screenSize[1])]

    def step(self,actionArg):

        stepReward = 0
        action = self.actions[actionArg]

        if action == 'UP':
            self.player = [self.player[0],self.player[1] - 1]
        elif action == 'DOWN':
            self.player = [self.player[0], self.player[1] + 1]
        elif action == 'LEFT':
            self.player = [self.player[0] - 1, self.player[1]]
        elif action == 'RIGHT':
            self.player = [self.player[0] + 1, self.player[1]]

        if self.player[0] < 0 or self.player[0] > self.screenSize[0] \
                or self.player[1] < 0 or self.player[1] > self.screenSize[1]:
            #stepReward -= 1
            self.died = True

        if self.player == self.apple:
            stepReward += 2
            self.apple = self.randomLocation()

        newState = [self.player[0],self.player[1],self.apple[0], self.apple[1]]
        self.totalReward += stepReward
        if self.render == True:
            self.updateDisplay()

        return newState, stepReward,self.died, {}

    def initialiseRender(self):
        self.RED = [255, 0, 0]
        self.GREEN = [0, 255, 0]
        pygame.init()
        self.screen = pygame.display.set_mode((self.screenSize[0]*10,self.screenSize[1]*10))
        pygame.mouse.set_visible(0)
        self.clock = pygame.time.Clock()

        player = pygame.Rect(self.player[0]*10, self.player[1]*10, 10, 10)

        apple = pygame.Rect(self.apple[0]*10, self.apple[1]*10, 10, 10)

        pygame.draw.rect(self.screen, self.RED, player)
        pygame.draw.rect(self.screen, self.GREEN, apple)

        pygame.display.flip()
        #self.clock.tick(60)

    def updateDisplay(self):
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.screen.fill((0, 0, 0))
        player = pygame.Rect(self.player[0]*10, self.player[1]*10, 10, 10)
        apple = pygame.Rect(self.apple[0]*10, self.apple[1]*10, 10, 10)
        pygame.draw.rect(self.screen, self.RED, player)
        pygame.draw.rect(self.screen, self.GREEN, apple)
        pygame.display.update()

        #self.clock.tick(60)






