# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.optimizers import Adam
from SnakeAI.SnakeEnv import SnakeEnvironment
import matplotlib.pyplot as plt
import timeit

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        # Hyperparameters
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.model_depth = 3
        self.layer_height = 64
        self.layers = []
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_shape = (self.state_shape)
        model = Sequential()
        model.add(Conv2D(4, (3, 3), padding='same', activation='relu',
                         input_shape=input_shape))
        model.add(MaxPool2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)
        start = timeit.default_timer()
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
        end = timeit.default_timer()
        print('replay time:', (end - start))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


EPISODES = 2000
DURATION = 500
SW = 30
SH = 30

if __name__ == "__main__":
    env = SnakeEnvironment(screenWidth = SW, screenHeight = SH)
    state_shape = env.state.shape
    action_size = len(env.actions)
    agent = DQNAgent(state_shape, action_size)
    batch_size = 10
    agent.load("snake-v2-dqn.h5")
    print(agent.model.summary())
    scores = []

    for e in range(EPISODES):

        env = SnakeEnvironment(screenWidth=SW, screenHeight=SH)
        state = env.state
        input_shape = (-1,) + state_shape
        state = state.reshape(input_shape)
        for time in range(DURATION+1):
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            end = timeit.timeit()

            next_state = state.reshape(input_shape)
            if reward != 0:
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or time == DURATION:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, env.totalReward, agent.epsilon))
                scores.append(env.totalReward)
                break

        if e % 10==0:
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.save("snake-v2-dqn.h5")




    plt.plot(range(len(scores)),scores)
    plt.title(('layer sizes: ',str(agent.layers),', network depth: ',str(len(agent.layers))))
    plt.show()