import argparse
from itertools import count

import gym
from gym.spaces import Box, Discrete
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    "A simple single layer MLP."

    def __init__(self, input_shape, output_size, hidden_size):
        super().__init__()

        self.flattened_input_size = 1
        for dim in input_shape:
            self.flattened_input_size *= dim

        self.fc1 = nn.Linear(self.flattened_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.flattened_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Policy:
    def __init__(self, net):
        self.net = net

    def select_action(self, state):
        "Returns action sampled from the policy distribution."
        state = torch.as_tensor(state, dtype=torch.float)
        logits = self.net(state.unsqueeze(0)).squeeze(0)
        return Categorical(logits=logits).sample().item()

    def log_probs(self, states, actions):
        "Returns log probabilities of the actions given states."
        logits = self.net(states)
        return Categorical(logits=logits).log_prob(actions)

def run_epsiode(policy, env, render=False, max_len=1000):
    "Runs one episodes according to the policy and returns the trajectory."
    states = []
    actions = []
    rewards = []

    state = env.reset()
    done = False

    while not done and len(states) < max_len:
        if render:
            env.render()
        states.append(state)
        action = policy.select_action(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)

    return states, actions, rewards

def collect_trajectories(policy, env, min_timesteps, gamma):
    "Returns trajectories as lists of states, actions, rewards and returns."
    states = []
    actions = []
    rewards = []
    returns = []

    for episode_num in count(1):
        episode_states, epsiode_actions, episode_rewards = run_epsiode(policy, env)
        episode_returns = calculate_returns(episode_rewards, gamma)

        states.extend(episode_states)
        actions.extend(epsiode_actions)
        rewards.extend(episode_rewards)
        returns.extend(episode_returns)

        if len(states) >= min_timesteps:
            break

    states = torch.as_tensor(np.stack(states), dtype=torch.float)
    actions = torch.as_tensor(actions, dtype=torch.long)
    rewards = torch.as_tensor(rewards, dtype=torch.float)
    returns = torch.as_tensor(returns, dtype=torch.float)

    return states, actions, rewards, returns

def calculate_returns(rewards, gamma):
    "Calcultes returns for given sequence of rewards."
    T = len(rewards)
    returns = [0] * T
    for t in range(T - 1, -1, -1):
        returns[t] = rewards[t] + gamma * (returns[t + 1] if t + 1 < T else 0)
    return returns

def optimiser_step(optimiser, loss):
    "Update paramaters corresponding to the optimiser."
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

def train(env_name, batch_size, num_updates, clip_ratio, hidden_size, gamma, policy_lr, value_fn_lr, test, render):
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), "State space must be continuos."
    assert isinstance(env.action_space, Discrete), "Action space must be discrete."

    policy_net = MLP(env.observation_space.shape, env.action_space.n, hidden_size)
    policy = Policy(policy_net)
    value_fn = MLP(env.observation_space.shape, 1, hidden_size)

    policy_optimiser = optim.Adam(policy.net.parameters(), lr=policy_lr)
    value_fn_optimiser = optim.Adam(value_fn.parameters(), lr=value_fn_lr)

    for i in count(1):
        if test:
            _, _, rewards, = run_epsiode(policy, env, render=render)
            print("Episode {}: {}".format(i, sum(rewards)))
            env.close()
            env = gym.make(env_name)

        # Collect trajectories and calculate rewards
        states, actions, rewards, returns = collect_trajectories(policy, env, batch_size, gamma)

        # Calculate advantage
        advantages = returns - value_fn(states).squeeze(1).detach()

        # Calculate initial log probabilities
        log_probs_old = policy.log_probs(states, actions).detach()

        for _ in range(num_updates):
            log_probs = policy.log_probs(states, actions)
            ratio_probs = torch.exp(log_probs - log_probs_old)
            clipped_prod = torch.clamp(ratio_probs, 1-clip_ratio, 1+clip_ratio) * advantages
            policy_loss = -torch.min(ratio_probs * advantages, clipped_prod).mean()
            optimiser_step(policy_optimiser, policy_loss)

        # Update value function to better match the returns
        value_fn_loss = F.mse_loss(value_fn(states).squeeze(1), returns)
        optimiser_step(value_fn_optimiser, value_fn_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v2", type=str)
    parser.add_argument("--batch_size", default=5000, type=int)
    parser.add_argument("--num_updates", default=80, type=int)
    parser.add_argument("--clip_ratio", default=0.2, type=float)
    parser.add_argument("--hidden_size", default=32, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--policy_lr", default=1e-2, type=float)
    parser.add_argument("--value_fn_lr", default=1e-2, type=float)
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--no_render", action="store_true")

    args = parser.parse_args()

    train(env_name=args.env,
          batch_size=args.batch_size,
          num_updates=args.num_updates,
          clip_ratio=args.clip_ratio,
          hidden_size=args.hidden_size,
          gamma=args.gamma,
          policy_lr=args.policy_lr,
          value_fn_lr=args.value_fn_lr,
          test=not args.no_test,
          render=not args.no_render)
