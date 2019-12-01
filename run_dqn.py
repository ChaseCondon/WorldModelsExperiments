import os
import gym
import vizdoomgym
import numpy as np
import skimage as skimage
from skimage import transform, color, exposure


from models.DQN import DQNAgent

NUM_STEPS     = 5000

EPSILON = 0.9
EPSILON_DECAY = 0.99

def preprocessImg(img, size):
    img = np.rollaxis(img, 0, 3)
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img)

    return img

if __name__ == '__main__':
    img_rows, img_cols = 64, 64
    # Convert image into Black and white
    img_channels = 4 # We stack 4 frames

    state_size = (img_rows, img_cols, img_channels)

    os.chdir('/nfs')
    env = gym.make('VizdoomTakeCover-v0')
    agent = DQNAgent(state_size, env.action_space.n)

    episode = 0
    total_reward = 0
    average_reward = 0
    last_average = 0
    episode_rewards = []
    logs = []
    total_t_step = 0

    print("Environment and Agent intialized. Beginning game...")
    # print(f"{os.getcwd()}/WorldModelsExperiments/models/ddqn.h5", os.path.exists(f"{os.getcwd()}/WorldModelsExperiments/models/ddqn.h5"))

    while True:

        episode += 1
        episode_reward = 0
        state = env.reset()
    
        for t_step in range(NUM_STEPS):
            total_t_step += 1
            x_t = preprocessImg(state, size=(img_rows, img_cols))
            s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
            s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4

            action_idx = agent.get_action(s_t)
            next_state, reward, done, info = env.step(action_idx)

            x_t1 = preprocessImg(next_state, size=(img_rows, img_cols))
            x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
            agent.replay_memory(s_t, action_idx, reward, s_t1, done, total_t_step)

            if total_t_step > agent.observe and total_t_step % agent.timestep_per_train == 0:
                Q_max, loss = agent.train_replay()
            
            state = next_state

            episode_reward += reward

            if t_step == NUM_STEPS or done:
                total_reward += episode_reward
                episode_rewards.append(episode_reward)
                break
        
        if episode%10 == 0:
            print(os.listdir(os.getcwd()))
            agent.save_model("dqn.h5")

        if len(episode_rewards) > 1:
            average_reward = np.mean(episode_rewards[:-1])
        
        if (episode-1)%50 == 0:
            print(f"Episode {episode}:\n\tlatest episode reward: {episode_reward}\n\ttotal episode reward: {total_reward}\n\taverage_reward: {average_reward}\n\tchange in average:{average_reward-last_average}")
        logs.append(f"Episode {episode}:\n\tlatest episode reward: {episode_reward}\n\ttotal episode reward: {total_reward}")

        if episode != 1 and abs(average_reward - last_average) < .01:
            break

        last_average = average_reward

    agent.save_model("dqn.h5")
    with open("run_dqn_out.txt", "w+") as file:
        for log in logs:
            file.write(log + '\n')