# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Dense, Input,Conv2D, Lambda, Flatten
from keras.backend import mean, max, expand_dims
from keras.optimizers import Adam
from SnakeAI.SnakeEnvConv import SnakeEnvironment
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.dueling_type = 'avg'

        # Hyperparameters
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.model_depth = 1
        self.layer_height = 2**5
        self.layers = []
        self.DQN = self._build_model()
        self.target_network = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        _input = Input(self.state_size)
        x = Conv2D(
        16,
        kernel_size=(3, 3),
        padding='same',
        strides=(1, 1),
        data_format='channels_first',
        activation='relu'
        )(_input)
        x = Conv2D(
            32,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            data_format='channels_first',
            activation='relu'
        )(x)
        x = Flatten()(x)
        x = Dense(256,  activation='relu')(x)
        y = Dense(self.action_size+1, activation='linear')(x)

        if self.dueling_type == 'avg':
            _output = Lambda(
                lambda a: expand_dims(a[:, 0], -1) + a[:, 1:] - mean(a[:, 1:], axis=1, keepdims=True),
                output_shape=(self.action_size,))(y)
        elif self.dueling_type == 'max':
            _output = Lambda(
                lambda a: expand_dims(a[:, 0], -1) + a[:, 1:] - max(a[:, 1:], axis=1, keepdims=True),
                output_shape=(self.action_size,))(y)
        else:
            _output = Lambda(lambda a: expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(self.action_size,))(y)


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


EPISODES = 1000
DURATION = 500
SW = 10
SH = 10
tau = 100

if __name__ == "__main__":
    render = True
    env = SnakeEnvironment(screenWidth = SW, screenHeight = SH,render = render)
    state_size = env.state.shape
    action_size = len(env.actions)
    agent = DQNAgent(state_size, action_size)
    batch_size = 50
    agent.load("snake-v4-conv-dqn.h5")
    print(agent.DQN.summary())
    scores = []

    input_size = (-1,) + state_size

    for e in range(EPISODES):
        env = SnakeEnvironment(screenWidth=SW, screenHeight=SH, render=render)
        state = env.state

        state = state.reshape(input_size)

        for time in range(DURATION+1):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(input_size)
            if reward != 0:
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or time == DURATION:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, env.totalReward, agent.epsilon))
                scores.append(env.totalReward)
                break
            if time % tau == 0:
                agent.update_target_weights()
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 10 == 0:
            agent.save("snake-v4-conv-dqn.h5")


    plt.plot(range(len(scores)),scores)
    plt.title(('layer sizes: ',str(agent.layers),', network depth: ',str(len(agent.layers))))
    plt.show()