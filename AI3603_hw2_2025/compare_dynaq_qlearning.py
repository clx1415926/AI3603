# -*- coding:utf-8 -*-
# 比较Dyna-Q和Q-Learning的收敛速度
import math, os, time, sys
import numpy as np
import random
import gym
from agent import Dyna_QAgent, QLearningAgent, SarsaAgent
import matplotlib.pyplot as plt

##############################################
# 参数配置区域 - 在这里修改参数以调优
##############################################
# 训练参数
NUM_EPISODES = 400       # 训练的总episode数
MAX_STEPS = 200          # 每个episode的最大步数
RANDOM_SEED = 0          # 随机种子

# Q-Learning和Dyna-Q的共同参数
LEARNING_RATE = 0.1      # 学习率 (learning rate)
GAMMA = 0.95             # 折扣因子 (discount factor)
INITIAL_EPSILON = 1.0    # 初始探索率 (initial exploration rate)
EPSILON_DECAY = 0.95     # 探索率衰减 (epsilon decay rate)
MIN_EPSILON = 0.01       # 最小探索率 (minimum epsilon)

# Dyna-Q特有参数
PLANNING_STEPS = 10      # Dyna-Q的规划步数 (建议范围: 10-200)
                         # 增加这个值可以加速Dyna-Q的收敛

# 其他参数
RENDER_EPISODES = False  # 是否渲染训练过程

# 可视化参数
MOVING_AVERAGE_WINDOW = 15  # 移动平均窗口大小
PRINT_INTERVAL = 50         # 打印训练信息的间隔
##############################################

def moving_average(data, window_size=MOVING_AVERAGE_WINDOW):
    """计算移动平均"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

##############################################
# 第一部分：训练Q-Learning（模拟cliff_walk_qlearning.py）
##############################################
print("=" * 60)
print("第一部分：训练 Q-Learning")
print("=" * 60)

# 构建环境
env = gym.make("CliffWalking-v0")
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
num_states = env.observation_space.n

# 设置随机种子
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED)

# 构建Q-Learning智能体
qlearning_agent = QLearningAgent(
    all_actions=all_actions,
    state_dim=num_states,
    alpha=LEARNING_RATE,  
    gamma=GAMMA,   
    epsilon=INITIAL_EPSILON,
    epsilon_decay=EPSILON_DECAY,   
    epsilon_min=MIN_EPSILON,     
)

qlearning_episode_rewards = []
qlearning_epsilon_values = []

# 开始训练Q-Learning
for episode in range(NUM_EPISODES):
    episode_reward = 0
    s = env.reset()
    
    for iter in range(MAX_STEPS):
        a = qlearning_agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        qlearning_agent.learn(s, a, r, s_)
        s = s_
        episode_reward += r
        if isdone:
            if RENDER_EPISODES:
                time.sleep(0.1)
            break
    
    qlearning_episode_rewards.append(episode_reward)
    qlearning_epsilon_values.append(qlearning_agent.epsilon)
    qlearning_agent.decay_epsilon()
    
    if (episode + 1) % PRINT_INTERVAL == 0:
        print(
            "episode:",
            episode + 1,
            "episode_reward:",
            episode_reward,
            "epsilon:",
            qlearning_agent.epsilon,
        )

print('\nQ-Learning training over\n')
env.close()

# 计算移动平均（不显示图表）
qlearning_avg_rewards = moving_average(qlearning_episode_rewards)

##############################################
# 第二部分：训练Dyna-Q（模拟cliff_walk_dyna_q.py）
##############################################
print("\n" + "=" * 60)
print("第二部分：训练 Dyna-Q")
print("=" * 60)

# 构建环境
env = gym.make("CliffWalking-v0")
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
num_states = env.observation_space.n

# 设置随机种子
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED)

# 构建Dyna-Q智能体
dynaq_agent = Dyna_QAgent(
    all_actions=all_actions,
    state_dim=num_states,
    alpha=LEARNING_RATE,  
    gamma=GAMMA,   
    epsilon=INITIAL_EPSILON,
    epsilon_decay=EPSILON_DECAY,   
    epsilon_min=MIN_EPSILON,
    n_planning_steps=PLANNING_STEPS,  # Dyna-Q的规划步数
)

dynaq_episode_rewards = []
dynaq_epsilon_values = []

# 开始训练Dyna-Q
for episode in range(NUM_EPISODES):
    episode_reward = 0
    s = env.reset()
    
    for iter in range(MAX_STEPS):
        a = dynaq_agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        dynaq_agent.learn(s, a, r, s_)
        s = s_
        episode_reward += r
        if isdone:
            if RENDER_EPISODES:
                time.sleep(0.1)
            break
    
    dynaq_episode_rewards.append(episode_reward)
    dynaq_epsilon_values.append(dynaq_agent.epsilon)
    dynaq_agent.decay_epsilon()
    
    if (episode + 1) % PRINT_INTERVAL == 0:
        print(
            "episode:",
            episode + 1,
            "episode_reward:",
            episode_reward,
            "epsilon:",
            dynaq_agent.epsilon,
        )

print('\nDyna-Q training over\n')
env.close()

# 计算移动平均（不显示图表）
dynaq_avg_rewards = moving_average(dynaq_episode_rewards)

##############################################
# 第三部分：训练SARSA
##############################################
print("\n" + "=" * 60)
print("第三部分：训练 SARSA")
print("=" * 60)

# 构建环境
env = gym.make("CliffWalking-v0")
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
num_states = env.observation_space.n

# 设置随机种子
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED)

# 构建SARSA智能体
sarsa_agent = SarsaAgent(
    all_actions=all_actions,
    state_dim=num_states,
    alpha=LEARNING_RATE,  
    gamma=GAMMA,   
    epsilon=INITIAL_EPSILON,
    epsilon_decay=EPSILON_DECAY,   
    epsilon_min=MIN_EPSILON,
)

sarsa_episode_rewards = []
sarsa_epsilon_values = []

# 开始训练SARSA
for episode in range(NUM_EPISODES):
    episode_reward = 0
    s = env.reset()
    a = sarsa_agent.choose_action(s)
    
    for iter in range(MAX_STEPS):
        s_, r, isdone, info = env.step(a)
        a_ = sarsa_agent.choose_action(s_)
        sarsa_agent.learn(s, a, r, s_, a_)
        s = s_
        a = a_
        episode_reward += r
        if isdone:
            break
    
    sarsa_episode_rewards.append(episode_reward)
    sarsa_epsilon_values.append(sarsa_agent.epsilon)
    sarsa_agent.decay_epsilon()
    
    if (episode + 1) % PRINT_INTERVAL == 0:
        print(
            "episode:",
            episode + 1,
            "episode_reward:",
            episode_reward,
            "epsilon:",
            sarsa_agent.epsilon,
        )

print('\nSARSA training over\n')
env.close()

# 计算移动平均（不显示图表）
sarsa_avg_rewards = moving_average(sarsa_episode_rewards)

##############################################
# 第四部分：整合对比
##############################################
print("\n" + "=" * 60)
print("第四部分：整合对比")
print("=" * 60)

# 额外平滑处理
def smooth_curve(data, window=5):
    """对数据进行额外的平滑处理"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='same')

# 过滤reward > -200的数据
def filter_rewards(rewards, threshold=-200):
    """只保留大于阈值的reward数据"""
    filtered = []
    indices = []
    for i, r in enumerate(rewards):
        if r > threshold:
            filtered.append(r)
            indices.append(i)
    return np.array(filtered), np.array(indices)

# 先平滑再过滤（增加平滑窗口让曲线更平）
qlearning_smoothed = smooth_curve(qlearning_avg_rewards, window=30)
dynaq_smoothed = smooth_curve(dynaq_avg_rewards, window=30)
sarsa_smoothed = smooth_curve(sarsa_avg_rewards, window=30)

qlearning_filtered, qlearning_indices = filter_rewards(qlearning_smoothed)
dynaq_filtered, dynaq_indices = filter_rewards(dynaq_smoothed)
sarsa_filtered, sarsa_indices = filter_rewards(sarsa_smoothed)

# 绘制对比图
plt.figure(figsize=(12, 6))
plt.suptitle('Three Algorithms Performance Comparison (Reward > -200)', fontsize=16, fontweight='bold')

# 左图: Epsilon Decay对比（保持完整）
plt.subplot(1, 2, 1)
plt.plot(qlearning_epsilon_values, color='blue', linewidth=2, label='Q-Learning')
plt.plot(dynaq_epsilon_values, color='red', linewidth=2, label='Dyna-Q')
plt.plot(sarsa_epsilon_values, color='green', linewidth=2, label='SARSA')
plt.title('Epsilon Decay Curve')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.legend()
plt.grid(True, alpha=0.3)

# 右图: Average Reward对比（只显示>-200的部分）
plt.subplot(1, 2, 2)
if len(qlearning_filtered) > 0:
    plt.plot(qlearning_indices, qlearning_filtered, color='blue', linewidth=2, label='Q-Learning')
if len(dynaq_filtered) > 0:
    plt.plot(dynaq_indices, dynaq_filtered, color='red', linewidth=2, label='Dyna-Q')
if len(sarsa_filtered) > 0:
    plt.plot(sarsa_indices, sarsa_filtered, color='green', linewidth=2, label='SARSA')
plt.title(f'Average Reward Curve (window={MOVING_AVERAGE_WINDOW}, Reward > -200)')
plt.xlabel('Episode (Windowed)')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印统计信息
print("\n=== 性能对比分析 ===")
print(f"最终平均奖励 (最后100个episode):")
print(f"  Q-Learning: {np.mean(qlearning_episode_rewards[-100:]):.2f}")
print(f"  Dyna-Q:     {np.mean(dynaq_episode_rewards[-100:]):.2f}")
print(f"  SARSA:      {np.mean(sarsa_episode_rewards[-100:]):.2f}")

print(f"\n最佳移动平均奖励:")
print(f"  Q-Learning: {np.max(qlearning_avg_rewards):.2f}")
print(f"  Dyna-Q:     {np.max(dynaq_avg_rewards):.2f}")
print(f"  SARSA:      {np.max(sarsa_avg_rewards):.2f}")

print(f"\n参数设置:")
print(f"  基础参数（三个算法相同）: learning_rate={LEARNING_RATE}, gamma={GAMMA}, initial_epsilon={INITIAL_EPSILON}, "
      f"epsilon_decay={EPSILON_DECAY}, min_epsilon={MIN_EPSILON}")
print(f"  Dyna-Q额外参数: planning_steps={PLANNING_STEPS}")
print(f"  训练参数: num_episodes={NUM_EPISODES}, max_steps={MAX_STEPS}")
print(f"\n  说明：Q-Learning (off-policy), SARSA (on-policy), Dyna-Q (model-based)")
