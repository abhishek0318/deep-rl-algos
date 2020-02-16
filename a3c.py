import argparse
import copy
from itertools import count
from time import sleep

import gym
from gym.spaces import Box, Discrete
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

gym.logger.set_level(40)

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

def test_(args, policy, T, max_len=1e5, sleep_time=30):
    "Test thread function: runs one epsiode, prints rewards and sleeps."
    env = gym.make(args.env_name)
    policy_net_th = MLP(env.observation_space.shape, env.action_space.n, args.hidden_size)
    policy_th = Policy(policy_net_th)
    env.close()

    while True:
        env = gym.make(args.env_name)
        policy_th.net.load_state_dict(policy.net.state_dict())
        rewards_sum = 0
        i = 0
        state = env.reset()
        done = False

        while not done and i < max_len:
            i += 1
            if not args.no_render:
                env.render()
            action = policy_th.select_action(state)
            state, reward, done, _ = env.step(action)
            rewards_sum += reward

        print("Timestep {}: {}".format(T.value, rewards_sum))
        env.close()
        sleep(sleep_time)

def copy_gradients(model1, model2):
    # Copy gradient of parameters from model2 to model1
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        param1._grad = param2.grad.clone().detach()

def optimiser_step(optimiser, loss):
    "Update paramaters corresponding to the optimiser."
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

def train(args):
    env = gym.make(args.env_name)
    assert isinstance(env.observation_space, Box), "State space must be continuos."
    assert isinstance(env.action_space, Discrete), "Action space must be discrete."

    policy_net = MLP(env.observation_space.shape, env.action_space.n, args.hidden_size)
    policy_net.share_memory()
    policy = Policy(policy_net)

    value_fn = MLP(env.observation_space.shape, 1, args.hidden_size)
    value_fn.share_memory()
    env.close()

    T = mp.Value('i', 0)

    processes = []

    if not args.no_test:
        p = mp.Process(target=test_, args=(args, policy, T))
        p.start()
        processes.append(p)

    for rank in range(1, args.num_processes + 1):
        p = mp.Process(target=train_, args=(args, policy, value_fn, T))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def train_(args, policy, value_fn, T):
    "Thread specific train function."
    env = gym.make(args.env_name)

    # Thread specific policy and value_fn
    policy_net_th = MLP(env.observation_space.shape, env.action_space.n, args.hidden_size)
    policy_th = Policy(policy_net_th)
    value_fn_th = MLP(env.observation_space.shape, 1, args.hidden_size)

    policy_optimiser = optim.Adam(policy.net.parameters(), lr=args.policy_lr)
    value_fn_optimiser = optim.Adam(value_fn.parameters(), lr=args.value_fn_lr)

    t = 1
    done = True

    while T.value < args.T_max:
        t_start = t

        # Synchronise thread specific parameters
        policy_th.net.load_state_dict(policy.net.state_dict())
        value_fn_th.load_state_dict(value_fn.state_dict())

        if done:
            state = env.reset()
            done = False

        # Collect trajectory
        states = []
        actions = []
        rewards = []

        while not done and t - t_start < args.update_freq:
            states.append(state)
            action = policy_th.select_action(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            rewards.append(reward / args.rescale_reward)
            t += 1
            T.value = T.value + 1

        # Calculate returns
        R = 0
        if not done:
            last_state = torch.as_tensor(state, dtype=torch.float)
            R = value_fn_th(last_state.unsqueeze(0)).squeeze(0).item()

        N = t - t_start
        returns = [0] * N
        for i in range(N - 1, -1, -1):
            R = rewards[i] + args.gamma * R
            returns[i] = R

        states = torch.as_tensor(np.stack(states), dtype=torch.float)
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float)
        returns = torch.as_tensor(returns, dtype=torch.float)

        # Calculate policy gradient
        advantages = returns - value_fn_th(states).squeeze(1).detach()
        log_probs = policy_th.log_probs(states, actions)
        policy_loss = - (log_probs * advantages).sum()

        # Calculate entropy
        probs = log_probs.exp()
        entropy = -probs * log_probs
        policy_loss -= args.beta * entropy.sum()

        # Update policy parameters
        policy_th.net.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy_th.net.parameters(), args.max_grad_norm)
        copy_gradients(policy.net, policy_th.net)
        policy_optimiser.step()

        # Update value function to better match the returns
        value_fn_loss = F.mse_loss(value_fn_th(states).squeeze(1), returns)
        value_fn_th.zero_grad()
        value_fn_loss.backward()
        nn.utils.clip_grad_norm_(value_fn_th.parameters(), args.max_grad_norm)
        copy_gradients(value_fn, value_fn_th)
        value_fn_optimiser.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="LunarLander-v2", type=str)
    parser.add_argument("--num_processes", default=4, type=int)
    parser.add_argument("--update_freq", default=5, type=int)
    parser.add_argument("--T_max", default=1e6, type=float)
    parser.add_argument("--beta", default=1e-2, type=float)
    parser.add_argument("--rescale_reward", default=10.0, type=float)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--max_grad_norm", default=50.0, type=float)
    parser.add_argument("--policy_lr", default=1e-3, type=float)
    parser.add_argument("--value_fn_lr", default=1e-3, type=float)
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--no_render", action="store_true")

    args = parser.parse_args()

    mp.set_start_method('spawn')

    train(args)
