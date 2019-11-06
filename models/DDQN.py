import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

from .ReplayBuffer import ReplayBuffer

class DDQNAgent(object):
    """The Double Deep Q-Network"""

    def __init__(self, observation_shape, action_size, batch_size=4):
        self.observation_shape = observation_shape
        self.action_size = action_size 
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(50000)
        self.buffer_min_size = 10000

        self.learning_rate = 0.001
        self.gamma = 0.75
        
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.target_update_counter = 0
        self.update_target_step = 5
        

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
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), 
                      metrics=['accuracy'])
        
        return model

    def train(self, terminal_state, step):
        if self.replay_memory.size < self.buffer_min_size:
            return
        mini_batch = self.replay_buffer.sample(self.batch_size)

        state_batch = [a[0] for a in mini_batch]
        n_state_batch = [a[4] for a in mini_batch]

        current_qs_list = self.model.predict(state_batch/255)
        next_qs = self.target_model.predict(n_state_batch/255)

        X = []
        y = []

        for index, (current_state, action, reward, done, n_state) in enumerate(mini_batch):
            if not done:
                max_future_q = np.max(next_qs[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X)/255, np.array(y), batch_size=self.batch_size, 
                        verbose=0, shuffle=False)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_step:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    def add_experience(self, s, a, r, d, s2):
        self.replay_buffer.add(s, a, r, d, s2) 
    
    def get_action(self, state):
        return self.model.predict(state.reshape(-1, *state.shape)/255)[0]

    