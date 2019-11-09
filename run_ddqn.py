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
    env = gym.make('VizdoomTakeCover-v0')
    agent = DDQNAgent(env.observation_space.shape, env.action_space.n, batch_size=32)

    total_reward = 0
    episode_rewards = []
    logs = []

    for e in range(NUM_EPISODES):

        state = env.reset()
        episode_reward = 0

        for t_step in range(NUM_STEPS):
            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(agent.get_action(state))

            EPSILON *= EPSILON_DECAY

            try:
                next_state, reward, done, info = env.step(action)
            except Exception as ex:
                print(f"Exception encountered in ep {e}: {ex}")
                logs.append(f"Exception encountered in ep {e}: {ex}")

            agent.add_experience(state, action, reward, done, next_state)
            agent.train(done, t_step)
            state = next_state

            episode_reward += reward

            if done:
                total_reward += episode_reward
                episode_rewards.append(episode_reward)
                break
        
        if e%1000 == 0:
            print(f"episode {e+1}/{NUM_EPISODES}:\n\tlatest episode reward: {episode_reward}\n\ttotal episode reward: {total_reward}")
            logs.append(f"episode {e+1}/{NUM_EPISODES}:\n\tlatest episode reward: {episode_reward}\n\ttotal episode reward: {total_reward}")

    with open("run_dqnn_out.txt", "w+") as file:
        for log in logs:
            file.write(log + '\n')