# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
import random

##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #


class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(
        self,
        all_actions,
        state_dim=48,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    ):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.state_dim = state_dim
        self.num_actions = len(all_actions)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros((state_dim, self.num_actions))

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.all_actions)
        else:
            action = np.argmax(self.q_table[observation, :])
        return action

    def learn(self, s, a, r, s_, a_):
        """learn from experience"""
        predict_q = self.q_table[s, a]
        target_q = r + self.gamma * self.q_table[s_, a_]
        self.q_table[s, a] += self.alpha * (target_q - predict_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(
        self,
        all_actions,
        state_dim=48,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    ):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.state_dim = state_dim
        self.num_actions = len(all_actions)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros((state_dim, self.num_actions))

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.all_actions)
        else:
            action = np.argmax(self.q_table[observation, :])
        return action

    def learn(self, s, a, r, s_):
        """learn from experience"""
        predict_q = self.q_table[s, a]
        target_q = r + self.gamma * np.max(self.q_table[s_, :])
        self.q_table[s, a] += self.alpha * (target_q - predict_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    ##### END CODING HERE #####


class Dyna_QAgent(object):
    ##### START CODING HERE #####
    def __init__(
        self,
        all_actions,
        state_dim=48,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        n_planning_steps=10, 
    ):
        self.all_actions = all_actions
        self.state_dim = state_dim
        self.num_actions = len(all_actions)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.n_planning_steps = n_planning_steps 

        self.q_table = np.zeros((state_dim, self.num_actions))
        
        self.model = {}

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.all_actions)
        else:
            action = np.argmax(self.q_table[observation, :])
        return action

    def q_learn(self, s, a, r, s_):
        predict_q = self.q_table[s, a]
        target_q = r + self.gamma * np.max(self.q_table[s_, :])
        self.q_table[s, a] += self.alpha * (target_q - predict_q)

    def learn(self,s, a, r, s_):
        """learn from experience"""
        predict_q = self.q_table[s, a]
        target_q = r + self.gamma * np.max(self.q_table[s_, :])
        self.q_table[s, a] += self.alpha * (target_q - predict_q)
        
        if s not in self.model:
            self.model[s] = {}
        self.model[s][a] = (r, s_)
        for _ in range(self.n_planning_steps):
            old_s = random.choice(list(self.model.keys()))
            old_a = random.choice(list(self.model[old_s].keys()))
            old_r, old_s_ = self.model[old_s][old_a]
            self.q_learn(old_s, old_a, old_r, old_s_)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    ##### END CODING HERE #####
