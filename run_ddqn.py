import argparse
import random
import os

import gym
import vizdoomgym
import numpy as np

from models.DDQN import DDQNAgent

NUM_EPISODES  = 1000000
NUM_STEPS     = 5000

EPSILON = 0.9
EPSILON_DECAY = 0.99

if __name__ == '__main__':
    os.chdir('/nfs')
    f = open("run_dqnn_out.txt", "w+")
    env = gym.make('VizdoomTakeCover-v0')
    agent = DDQNAgent(env.observation_space.shape, env.action_space.n, batch_size=32)

    total_reward = 0
    episode_rewards = []

    for e in range(NUM_EPISODES):

        state = env.reset()
        episode_reward = 0

        for t_step in range(NUM_STEPS):
            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(agent.get_action(state))

            EPSILON *= EPSILON_DECAY

            next_state, reward, done, info = env.step(action)

            agent.add_experience(state, action, reward, done, next_state)
            agent.train(done, t_step)
            state = next_state

            episode_reward += reward

            if done:
                total_reward += episode_reward
                episode_rewards.append(episode_reward)
                break
        
        if e%1000 == 0:
            print(f"episode {e}/{NUM_EPISODES}:\n\tlatest episode reward: {episode_reward}\n\ttotal episode reward: {total_reward}")
            f.write(f"episode {e}/{NUM_EPISODES}:\n\tlatest episode reward: {episode_reward}\n\ttotal episode reward: {total_reward}")
    
    f.close()
            