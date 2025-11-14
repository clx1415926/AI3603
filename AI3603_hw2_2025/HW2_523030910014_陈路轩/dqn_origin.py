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
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.6,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=0.3,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
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

    the neural network model for approximating Q value function

    Inputs:
        State
    Outputs:
        Q-values for each possible action in that state.

    For LunarLander-v2:
    - Input layer: 8 (state dimension)
    - Hidden layer 1: 120 neurons, ReLU activation
    - Hidden layer 2: 84 neurons, ReLU activation
    - Output layer: 4 (action dimension)
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

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
            """
            optimizer.zero_grad()
            loss.backward()
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