# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Dense, Activation, Input, Lambda
from keras.backend import mean, max, expand_dims
from keras.optimizers import Adam
from SnakeAI.SnakeEnv import SnakeEnvironment
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.dueling_type = 'avg'

        # Hyperparameters
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.DQN = self._build_model()
        self.target_network = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        _input = Input(shape=(self.state_size,))
        x = Dense(12,  activation='relu')(_input)
        x = Dense(24, activation='relu')(x)

        #Dueling DQN
        x = Dense(self.action_size+1, activation='linear')(x)

        if self.dueling_type == 'avg':
            y = Lambda(
                lambda a: expand_dims(a[:, 0], -1) + a[:, 1:] - mean(a[:, 1:], axis=1, keepdims=True),
                output_shape=(self.action_size,))(x)
        elif self.dueling_type == 'max':
            y = Lambda(
                lambda a: expand_dims(a[:, 0], -1) + a[:, 1:] - max(a[:, 1:], axis=1, keepdims=True),
                output_shape=(self.action_size,))(x)
        else:
            y = Lambda(lambda a: expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(self.action_size,))(x)

        _output = Activation('softmax')(y)
        model = Model(inputs=_input,outputs=_output)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.DQN.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                #Double DQN
                action_for_next_state = self.act(next_state)
                target = (reward + self.gamma * self.target_network.predict(next_state)[0][action_for_next_state]
                          )
            target_f = self.DQN.predict(state)
            target_f[0][action] = target
            self.DQN.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.DQN.load_weights(name)

    def save(self, name):
        self.DQN.save_weights(name)

    #Fixed Q targets
    def update_target_weights(self):
        self.target_network.set_weights(self.DQN.get_weights())

EPISODES = 500
DURATION = 200
SS = 20
tau = 100

if __name__ == "__main__":
    render = True
    env = SnakeEnvironment(screenSize = SS,render = render)
    state_size = len(env.state)
    action_size = len(env.actions)
    agent = DQNAgent(state_size, action_size)
    batch_size = 50
    agent.load("snake-v4-dqn.h5")
    print(agent.DQN.summary())
    scores = []

    for e in range(EPISODES):
        env = SnakeEnvironment(screenSize = SS, render=render)
        state = env.state
        state = np.reshape(state, [1, state_size])
        apples_collected = 0
        for time in range(DURATION+1):
            action = agent.act(state)
            next_state, reward, done, apple = env.step(action)
            apples_collected += apple
            next_state = np.reshape(next_state, [1, state_size])
            if time == DURATION:
                reward = -10
            if reward != 0:
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or time == DURATION:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, apples_collected, agent.epsilon))
                scores.append(env.totalReward)
                break
            if time % tau == 0:
                agent.update_target_weights()

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

        if e % 10 == 0:
            agent.save("snake-v4-dqn.h5")


    plt.plot(range(len(scores)),scores)
    plt.show()