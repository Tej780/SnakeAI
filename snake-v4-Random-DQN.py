# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from SnakeAI.SnakeEnv import SnakeEnvironment
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        # Hyperparameters
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.model_depth = random.randint(1,5)
        self.layer_height = 2**random.randint(5,7)
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        for i in range(self.model_depth):
            model.add(Dense(self.layer_height, activation='relu'))
            model.add(Dropout(0.3))
        model.add(Dense(self.action_size, activation='linear'))
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
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


EPISODES = 50
DURATION = 500
SW = 20
SH = 20

if __name__ == "__main__":
    render = True
    env = SnakeEnvironment(screenWidth = SW, screenHeight = SH,render = render)
    state_size = len(env.state)
    action_size = len(env.actions)
    agent = DQNAgent(state_size, action_size)
    batch_size = 50
    #agent.load("snake-v1-dqn.h5")
    print(agent.model.summary())
    scores = []

    for e in range(EPISODES):
        env = SnakeEnvironment(screenWidth=SW, screenHeight=SH, render=render)
        state = env.state
        state = np.reshape(state, [1, state_size])
        for time in range(DURATION+1):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            if reward != 0:
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or time == DURATION:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, env.totalReward, agent.epsilon))
                scores.append(env.totalReward)
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.save("snake-v1-dqn.h5")


    plt.plot(range(len(scores)),scores)
    plt.title(('layer sizes: ',str(agent.layer_height),', network depth: ',str(agent.model_depth)))
    plt.xlabel('No. episodes')
    plt.ylabel('Score')
    plt.show()
    plt.savefig('score_trend.png')