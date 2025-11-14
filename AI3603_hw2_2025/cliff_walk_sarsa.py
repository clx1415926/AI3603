# -*- coding:utf-8 -*-
# Train Sarsa in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import SarsaAgent

##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
# get the size of action space
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

####### START CODING HERE #######
num_states = env.observation_space.n

# construct the intelligent agent.
agent = SarsaAgent(
    all_actions=all_actions,
    state_dim=num_states,
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_decay=0.99,
    epsilon_min=0.01,
)

episode_rewards = []
epsilon_values = []

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # agent interacts with the environment
    a = agent.choose_action(s)
    for iter in range(500):
        s_, r, isdone, info = env.step(a)
        a_ = agent.choose_action(s_)
        agent.learn(s, a, r, s_, a_)
        s = s_
        a = a_
        episode_reward += r
        if isdone:
            # time.sleep(0.1)
            break
    episode_rewards.append(episode_reward)
    epsilon_values.append(agent.epsilon)
    agent.decay_epsilon()

    if (episode + 1) % 50 == 0:
        print(
            "episode:",
            episode + 1,
            "episode_reward:",
            episode_reward,
            "epsilon:",
            agent.epsilon,
        )

print("\ntraining over\n")

# close the render window after training.
env.close()




import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.suptitle('SARSA Algorithm Performance', fontsize=16, fontweight='bold')
plt.subplot(1, 2, 1) 
plt.plot(epsilon_values)
plt.title('Epsilon Decay Curve')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

def moving_average(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

avg_rewards = moving_average(episode_rewards)
plt.subplot(1, 2, 2) 
plt.plot(avg_rewards)
plt.title(f'Average Reward Curve (window={20})')
plt.xlabel('Episode (Windowed)')
plt.ylabel('Average Reward')

plt.tight_layout()
plt.show() 



s = env.reset()
agent.epsilon = 0.0 
episode_reward = 0
isdone = False
path = [s]  # 记录路径

while not isdone:
    env.render() 
    time.sleep(0.3) 
    a = agent.choose_action(s) 
    s, r, isdone, info = env.step(a)
    path.append(s)
    episode_reward += r

print(f"Test complete! Final path reward: {episode_reward}")

# 可视化路径（在环境图上画红线）
img = env.render(mode='rgb_array')
env.close()
plt.figure(figsize=(12, 4))
plt.imshow(img)
coords = [(divmod(s, 12)[1] * img.shape[1] / 12 + img.shape[1] / 24, 
           divmod(s, 12)[0] * img.shape[0] / 4 + img.shape[0] / 8) for s in path]
plt.plot([c[0] for c in coords], [c[1] for c in coords], 'r-', linewidth=3, marker='o', markersize=5)
plt.title(f'SARSA Path (Reward: {episode_reward})', fontsize=14, fontweight='bold')
plt.axis('off')
plt.savefig('sarsa_path.png', dpi=150, bbox_inches='tight')
plt.show()
####### END CODING HERE #######
