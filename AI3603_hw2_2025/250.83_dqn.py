# -*- coding:utf-8 -*-
import argparse
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=100000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=600,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1.0,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.02,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.20,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--max-grad-norm", type=float, default=10.0,
        help="the maximum norm for gradient clipping")
    args = parser.parse_args()
    args.env_id = "LunarLander-v2"
    return args

def make_env(env_id, seed):
    """construct the gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class QNetwork(nn.Module):
    """
    comments: 

    Improved Q-network with deeper architecture and regularization.

    Inputs:
        State
    Outputs:
        Q-values for each possible action in that state.

    For LunarLander-v2:
    - Input layer: 8 (state dimension)
    - Hidden layer 1: 256 neurons, ReLU activation
    - Hidden layer 2: 256 neurons, ReLU activation  
    - Hidden layer 3: 128 neurons, ReLU activation
    - Output layer: 4 (action dimension)
    - Dropout for regularization to prevent overfitting
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
        )
        
        # Better weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """
    comments: 
    
    Implements a linear decay for epsilon (ε) as part of the ε-greedy strategy.
    - start_e: The initial value of epsilon
    - end_e: The final value of epsilon 
    - duration: The total number of timesteps over which to decay from start_e to end_e.
    - t: The current timestep.

    When t >= duration, epsilon will be equal to end_e.
    When t < duration, epsilon will decrease linearly from start_e to end_e.
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def exponential_schedule(start_e: float, end_e: float, duration: int, t: int):
    """
    Implements an exponential decay for epsilon.
    Provides faster initial decay and slower later decay, which can lead to better exploration.
    """
    if t >= duration:
        return end_e
    # Calculate decay rate such that we reach end_e at duration
    decay_rate = -np.log(end_e / start_e) / duration
    return max(start_e * np.exp(-decay_rate * t), end_e)

if __name__ == "__main__":
    
    """parse the arguments"""
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    """we utilize tensorboard yo log the training process"""
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    """
    comments:
    set the random seed to make the experiment reproducible(for numpy, torch, gym, random)
    check whether cuda is available, if available, use cuda to accelerate the training, else use cpu
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    comments: 
    create the gym environment
    envs: the vectorized environment
    """
    envs = make_env(args.env_id, args.seed)

    """
    comments: 
    Initialize the Q-network, target network, and optimizer.

    The target network is a copy of the Q-network and is used to stabilize training.
    - q_network (Main): Used for action selection (exploitation) and is the network that gets updated by the optimizer.
    - target_network: Used to calculate the TD Target value. Its weights are frozen.It stabilizes training.
    - Adam optimizer: Train the parameters of the q_network.
    """
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """
    comments: 
    Initialize the Replay Buffer.
    - DQN uses a replay buffer to store past experiences (s, a, r, s', d).
    Sampling random batches from the buffer breaks the correlation between consecutive samples, stabilizing training.
    """
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    """
    comments: 
    Initialize the environment by resetting it and getting the first observation (state).
    """
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        
        """
        comments: 
        Calculate the current value of Epsilon.
        Uses the linear_schedule function based on the current global_step to determine the probability of exploration.
        """
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        """
        comments: 
        Epsilon-Greedy (ε-greedy) Action Selection.
        - With probability 'epsilon', choose a random action (exploration).
        - With probability '1-epsilon', choose the action with the highest predicted Q-value from the main q_network (exploitation).
        """
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
        
        """
        comments: 
        Interact with the environment.
        Inputs:
            - actions: The actions chosen by the agent based on the ε-greedy strategy.
        Outputs:
            - next_obs (s'): The next state.
            - rewards (r): The immediate reward.
            - dones (d): whether the episode has ended.
            - infos: extra info .
        """
        next_obs, rewards, dones, infos = envs.step(actions)
        # envs.render() # close render during training
        
        if dones:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
        
        """
        comments: 
        Store the transition (s, a, r, s', d) in the replay buffer.
        """
        rb.add(obs, next_obs, actions, rewards, dones, infos)
        
        """
        comments: 
        Update the current observation to the next observation.
        """
        obs = next_obs if not dones else envs.reset()
        
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            
            """
            comments: 
            Sample a random batch of experiences from the Replay Buffer.
            """
            data = rb.sample(args.batch_size)
            
            """
            comments:
            Calculate the TD Target value (y_j).
            y_j = r + γ * max_a' Q_target(s', a')   (if not done)
            y_j = r                               (if done)

            """
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            """
            comments: 
            Log the loss and average Q-value to TensorBoard to monitor the training process.
            """
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            
            """
            comments: 
            Perform backpropagation and update the network.
            Apply gradient clipping to prevent gradient explosion.
            """
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
            optimizer.step()
            
            """
            comments: 
            Periodically update the target network to match the weights of the main q_network.
            """
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
    
    """close the env and tensorboard logger"""
    envs.close()
    writer.close()


# ==================================================================================
    # ##### START OF ADDED CODE (Homework 2, Section 2.3) #####
    #
    # This section is added to fulfill the requirements of:
    # 1. Show the training process (how to view the learning curve).
    # 2. Demonstrate the final training result (run the agent).
    # 3. Provide the training result (save the model and a video ).
    # ==================================================================================
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Run data saved in: runs/{run_name}")
    print("="*60 + "\n")

    # --- 1. Training Process (Learning Curve)  ---
    print(f"--- 1. 训练过程 (Learning Curve) ---")
    print("训练过程中的 Episodic Return 已被记录到 TensorBoard。")
    print("请在你的终端中运行以下命令来查看学习曲线: \n")
    print(f"    tensorboard --logdir=runs/{run_name}\n")

    # --- 2. Save Final Model (Final Training Effect) ---
    print(f"--- 2. 保存最终模型 ---")
    model_path = f"runs/{run_name}/{args.exp_name}.pth"
    torch.save(q_network.state_dict(), model_path)
    print(f"最终模型权重已保存至: {model_path}\n")

    # --- 3. Run Demonstration and Record Video (Final Training Effect)  ---
    print(f"--- 3. 运行最终效果演示 (并录制视频) ---")
    
    # 按照 PDF  建议, 我们使用 gym.wrappers.RecordVideo
    # RecordVideo 需要环境在 'rgb_array' 模式下渲染
    video_folder = f"runs/{run_name}/videos"
    
    # 创建一个新的环境用于演示
    # 注意: 我们需要手动创建并设置 render_mode="rgb_array" 以便 RecordVideo 正常工作
    demo_env = gym.make(args.env_id, render_mode="rgb_array")
    
    # 添加 PDF  中建议的 wrapper
    # episode_trigger=lambda e: True 表示录制所有评估回合
    demo_env = gym.wrappers.RecordVideo(demo_env, video_folder, episode_trigger=lambda e: True)
    # 我们也添加 RecordEpisodeStatistics 来方便地追踪回报
    demo_env = gym.wrappers.RecordEpisodeStatistics(demo_env)

    print(f"视频将保存至: {video_folder}")
    
    num_demo_episodes = 5  # 演示 5 个回合
    demo_returns = []  # 存储演示回合的得分用于统计
    
    for i in range(num_demo_episodes):
        obs = demo_env.reset()
        done = False
        while not done:
            # 评估时，我们不再使用 epsilon-greedy，而是直接选择最优动作
            with torch.no_grad():
                q_values = q_network(torch.Tensor(obs).to(device))
                action = torch.argmax(q_values, dim=0).cpu().numpy()
            
            # 与环境交互
            obs, reward, done, info = demo_env.step(action)
            
            if done:
                # RecordEpisodeStatistics 会在 info['episode'] 中记录回报
                episode_return = info['episode']['r']
                demo_returns.append(episode_return)
                print(f"  演示回合 {i+1} 结束, Episodic Return: {episode_return:.2f}")

    # 关闭演示环境
    demo_env.close()
    print(f"\n演示完成。演示回合平均得分: {np.mean(demo_returns):.2f}\n")
    
    # --- 4. 测试回合评估 (Test Episodes for Performance Evaluation) ---
    print("="*60)
    print(f"--- 4. 测试回合评估 (200回合) ---")
    print("开始运行200个测试回合以全面评估性能...")
    
    # 创建测试环境（不录制视频）
    test_env = gym.make(args.env_id)
    test_env = gym.wrappers.RecordEpisodeStatistics(test_env)
    
    num_test_episodes = 200  # 测试 200 个回合
    test_returns = []  # 存储测试回合的得分
    
    for i in range(num_test_episodes):
        obs = test_env.reset()
        done = False
        while not done:
            # 使用贪婪策略（不探索）
            with torch.no_grad():
                q_values = q_network(torch.Tensor(obs).to(device))
                action = torch.argmax(q_values, dim=0).cpu().numpy()
            
            obs, reward, done, info = test_env.step(action)
            
            if done:
                episode_return = info['episode']['r']
                test_returns.append(episode_return)
                # 每20个回合打印一次进度
                if (i + 1) % 20 == 0:
                    current_avg = np.mean(test_returns)
                    print(f"  已完成 {i+1}/{num_test_episodes} 个测试回合, 当前平均得分: {current_avg:.2f}")
    
    # 关闭测试环境
    test_env.close()
    
    # --- 5. 综合性能评估统计 (Comprehensive Performance Evaluation) ---
    print("\n" + "="*60)
    print("--- 5. 综合性能评估统计 (Comprehensive Performance Evaluation) ---")
    
    # 演示回合统计
    print(f"\n【演示回合统计】(用于视频展示, {len(demo_returns)}个回合)")
    print(f"  平均得分: {np.mean(demo_returns):.2f} ± {np.std(demo_returns):.2f}")
    print(f"  得分范围: [{np.min(demo_returns):.2f}, {np.max(demo_returns):.2f}]")
    
    # 测试回合统计
    print(f"\n【测试回合统计】(用于性能评估, {len(test_returns)}个回合)")
    print(f"  平均得分: {np.mean(test_returns):.2f} ± {np.std(test_returns):.2f}")
    print(f"  最高得分: {np.max(test_returns):.2f}")
    print(f"  最低得分: {np.min(test_returns):.2f}")
    print(f"  中位数得分: {np.median(test_returns):.2f}")
    print(f"  得分范围: [{np.min(test_returns):.2f}, {np.max(test_returns):.2f}]")
    
    # 成功率统计（得分>=200视为成功）
    success_rate = np.sum(np.array(test_returns) >= 200) / len(test_returns) * 100
    print(f"  成功率 (得分≥200): {success_rate:.1f}% ({np.sum(np.array(test_returns) >= 200)}/{len(test_returns)})")
    
    # 性能评级（基于测试回合）
    test_avg_score = np.mean(test_returns)
    print(f"\n【最终性能评级】")
    if test_avg_score >= 200:
        print(f"  ✅ 优秀 (Excellent) - 测试平均得分 {test_avg_score:.2f} >= 200")
    elif test_avg_score >= 100:
        print(f"  ⚠️  良好 (Good) - 测试平均得分 {test_avg_score:.2f} >= 100")
    elif test_avg_score >= 0:
        print(f"  ⚠️  及格 (Pass) - 测试平均得分 {test_avg_score:.2f} >= 0")
    else:
        print(f"  ❌ 需要改进 (Needs Improvement) - 测试平均得分 {test_avg_score:.2f} < 0")
    
    print("="*60)
    
    # ##### END OF ADDED CODE #####
    # ==================================================================================