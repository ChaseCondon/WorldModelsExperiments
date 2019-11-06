import argparse

import gym
import vizdoomgym

from models.DQN import DQNAgent

NUM_EPISODES  = 1000000
NUM_STEPS     = 5000

def get_agents(m, env):
    agents = []

    m = m.lower()

    if m not in ['dqn', 'ddqn', 'a3c', 'rainbow', 'worldmodel']:
        m = 'all'

    if m is 'dqn' or m is 'all':
        agents.append(DQNAgent(env.observation_space.shape, env.action_space.n))

    return agents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline runner for RL models.')
    parser.add_argument('-m', default='all', type=str, help='Specify models to run (default: all)')
    parser.add_argument('-o', type=str, help='Output directory for the report')

    args = parser.parse_args()

    env = gym.make('VizdoomBasic-v0')
    agents = get_agents(args.m, env)

    for agent in agents:
        total_reward = 0
        episode_rewards = []

        for e in range(NUM_EPISODES):

            state = env.reset()
            episode_reward = 0

            for t_step in range(NUM_STEPS):
                action = agent.get_action(state)

                next_state, reward, done, info = env.step(action)

                episode_reward += reward

                if done:
                    total_reward += episode_reward
                    episode_rewards.append(episode_reward)
                    break
                