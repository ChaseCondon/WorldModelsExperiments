import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

from .ReplayBuffer import ReplayBuffer

class DQNAgent(Agent):
    """The Deep Q-Network"""

    def __init__(self, observation_shape, action_size, batch_size=4):
        self.observation_shape = observation_shape
        self.action_size = action_size 
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(10000)

        self.gamma = 0.75
        self.epsilon = 0.1
        self.epsilon_decay = 0.99

        self.model = self.create_model()
        

    def create_model(self):
        """Creates the model in Tensorflow 2 Keras"""
        model = Sequential()

        model.add(Conv2D(64, (4, 4), input_shape=self.observation_shape, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, (4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
of task, or at least one that can be represented visually. Up til now, we've really only been visualizing the environment for our benefit. 
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), 
                      metrics=['accuracy'])
        
        return model

    def get_action(self, state):
        return 0