# based on code from: https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/ddqn.py
import random

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
import numpy as np

from .ReplayBuffer import ReplayBuffer

class DoubleDQNAgent():
    """The Double Deep Q-Network"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size 

        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 32
        self.observe = 5000
        self.explore = 50000
        self.frame_per_action = 4
        self.update_target_freq = 3000
        self.timestep_per_train = 100 # number of timesteps between training interval
        
        self.replay_buffer = ReplayBuffer(50000)
        self.buffer_min_size = 10000

        self.model = self.create_model(self.state_size, self.action_size, self.learning_rate)
        self.target_model = self.create_model(self.state_size, self.action_size, self.learning_rate)

    def create_model(self, input_shape, action_size, learning_rate):
        model = keras.Sequential()
        model.add(kl.Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                                    input_shape=(input_shape)))
        model.add(kl.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(kl.Conv2D(54, (3, 3), activation='relu'))
        model.add(kl.Flatten())
        model.add(kl.Dense(512, activation='relu'))
        model.add(kl.Dense(action_size, activation='linear'))

        adam = tf.keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model
    
    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """

        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            q = self.model.predict(state)
            action_idx = np.argmax(q)
        return action_idx
           
    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated, t):
        self.replay_buffer.add(s_t, action_idx, r_t, is_terminated, s_t1)
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
        
        if t % self.update_target_freq == 0:
            self.update_target_model()
    
    def train_minibatch_replay(self):
        """
        Train on a single minibatch
        """
        mini_batch = self.replay_buffer.sample(self.batch_size)
    
        update_input = np.zeros(((batch_size,) + self.state_size)) # Shape 64, img_rows, img_cols, 4
        update_target = np.zeros(((batch_size,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i,:,:,:] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i,:,:,:] = mini_batch[i][4]
            done.append(mini_batch[i][3])

        target = self.model.predict(update_input) # shape 64, num_actions

        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])
        
        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        loss = self.model.train_on_batch(update_input, target)

        return np.max(target[-1]), loss

    # pick samples randomly from replay memory (with batch size)
    def train_replay(self):

        num_samples = min(self.batch_size * self.timestep_per_train, self.replay_buffer.size)
        replay_samples = self.replay_buffer.sample(num_samples)

        update_input = np.zeros(((num_samples,) + self.state_size))
        update_target = np.zeros(((num_samples,) + self.state_size))
        action, reward, done = [], [], []

        for i in range(num_samples):
            update_input[i,:,:,:] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            update_target[i,:,:,:] = replay_samples[i][4]
            done.append(replay_samples[i][3])

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)

        for i in range(num_samples):
            # like Q learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])

        loss = self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)

        return np.max(target[-1]), loss.history['loss']

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)
    
    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name, overwrite=True)