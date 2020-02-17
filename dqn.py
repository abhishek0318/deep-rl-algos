import argparse
from collections import deque
from datetime import datetime
from itertools import count
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage import color
from skimage.transform import resize

class DeepQNet:
    def __init__(self, env_name, hidden_size):
        self.env_name = env_name
        env = gym.make(self.env_name)
        self.action_space = env.action_space
        self.net_args = (env.observation_space.shape, env.action_space.n, hidden_size)
        self.net = self.Net(*self.net_args)

    class Net(nn.Module):
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

    class ReplayMemory:
        # Code taken from https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3

        def __init__(self, max_size):
            max_size = int(max_size)
            self.buffer = [None] * max_size
            self.max_size = max_size
            self.index = 0
            self.size = 0

        def append(self, obj):
            self.buffer[self.index] = obj
            self.size = min(self.size + 1, self.max_size)
            self.index = (self.index + 1) % self.max_size

        def sample(self, batch_size):
            sample_size = min(batch_size, self.size)
            indices = random.sample(range(self.size), sample_size)
            return [self.buffer[index] for index in indices]

    def random_action(self):
        return self.action_space.sample()

    def best_action(self, state):
        device = next(self.net.parameters()).device
        state = state.to(device)
        return self.net(state.unsqueeze(0)).squeeze(0).argmax().item()

    def update_net(self, batch, target_net, optimizer, discount_factor):
        # batch: [(state, action, reward, new_state, finished)] * batch_size

        device = next(self.net.parameters()).device
        state = torch.stack([x[0] for x in batch], dim=0).to(device)
        action = torch.LongTensor([x[1] for x in batch]).to(device)
        reward = torch.FloatTensor([x[2] for x in batch]).to(device)
        new_state = torch.stack([x[3] for x in batch], dim=0).to(device)
        finished = torch.FloatTensor([x[4] for x in batch]).to(device)

        max_q = target_net(new_state).max(dim=1)[0]
        mask = 1 - finished
        max_q *= mask
        target = reward + discount_factor * max_q
        target.detach_()

        q = self.net(state)[range(target.shape[0]), action]

        loss = (target - q) ** 2
        loss.clamp_(min=-1, max=1)
        loss = loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def initialise_replay_memory(self, replay_memory, replay_start_size):
        pbar = tqdm(total=replay_start_size)

        env = gym.make(self.env_name)
        for episode in count(0):
            state = env.reset()
            state = torch.FloatTensor(state)

            finished = False
            while not finished:
                pbar.update()

                action = self.random_action()
                new_state, reward, finished, _ = env.step(action)

                new_state = torch.FloatTensor(new_state)
                replay_memory.append((state, action, reward / 10, new_state, finished))
                state = new_state.clone().detach()

                if replay_memory.size == replay_start_size:
                    pbar.close()
                    env.close()
                    return

    def linear_decay(self, frame_number, final_exploration_frame=1000000, initial_exploration=1,
                     final_exploration=0.1):

        if frame_number > final_exploration_frame:
            return final_exploration
        else:
            difference = initial_exploration - final_exploration
            return initial_exploration - difference * ((frame_number - 1) / final_exploration_frame)

    def train(self, training_frames, minibatch_size, replay_memory_size, target_network_update_frequency,
              discount_factor, learning_rate, initial_exploration, final_exploration, final_exploration_frame,
              replay_start_size, test, render):

        # initialise replay memory
        replay_memory = self.ReplayMemory(replay_memory_size)
        self.initialise_replay_memory(replay_memory, replay_start_size)

        # intialise action value function q with random weights
        self.net.train()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(device)

        # initialise target action value function
        target_net = self.Net(*self.net_args)
        target_net.to(device)
        target_net.load_state_dict(self.net.state_dict())

        # initialise environmnent and optimiser
        env = gym.make(self.env_name)
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        epsilon_fn = lambda x: self.linear_decay(x, final_exploration_frame,
                                                 initial_exploration, final_exploration)

        # initialise logging related things
        pbar = tqdm(total=training_frames)
        test_reward = 0

        # initialise counts
        frame_number = 0

        for episode in count(1):
            # initialise statistics for logging
            episode_reward = 0
            episode_loss = 0
            episode_frames = 0

            # initialise sequence
            state = env.reset()
            state = torch.FloatTensor(state)

            finished = False
            while not finished:
                episode_frames += 1
                frame_number += 1
                pbar.update()

                # select random action with epsilon probability else select best action
                epsilon = epsilon_fn(frame_number)
                action = self.best_action(state) if random.random() > epsilon else self.random_action()
                # execute action in emulator and obtain next image, reward
                new_state, reward, finished, _ = env.step(action)

                new_state = torch.FloatTensor(new_state)
                # store transition in replay_memory
                replay_memory.append((state, action, reward / 10, new_state, finished))
                state = new_state.clone().detach()

                # Sample batch from replay memory and update parameters
                batch = replay_memory.sample(minibatch_size)
                loss = self.update_net(batch, target_net, optimizer, discount_factor)

                # update the target net after target_network_update_frequency steps
                if frame_number % target_network_update_frequency == 0:
                    target_net.load_state_dict(self.net.state_dict())

                # maintain statistics for logging
                episode_reward += reward
                episode_loss += loss

                # stop training
                if frame_number == training_frames:
                    pbar.close()
                    env.close()
                    return

            if episode % 50 == 0 and test:
                test_reward = self.play(render=render)
                self.net.train()
                self.net.to(device)
            pbar.set_postfix(episode=episode, reward=episode_reward, loss=episode_loss / episode_frames,
                             length=episode_frames, test_reward=test_reward)

    def play(self, render=True):
        self.net.eval()
        self.net.to("cpu")
        env = gym.make(self.env_name)

        reward_sum = 0
        state = env.reset()
        finished = False
        while not finished:
            if render:
                env.render()
            state = torch.FloatTensor(state)
            action = self.best_action(state)
            state, reward, finished, _ = env.step(action)
            reward_sum += reward
        env.close()
        return reward_sum

    def save(self, path):
        self.net.to("cpu")
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.to("cpu")
        self.net.load_state_dict(torch.load(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v2", type=str)
    parser.add_argument("--training_frames", default=400000, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--minibatch_size", default=32, type=int)
    parser.add_argument("--replay_memory_size", default=200000, type=int)
    parser.add_argument("--target_network_update_frequency", default=10000, type=int)
    parser.add_argument("--discount_factor", default=0.99, type=float)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--initial_exploration", default=1, type=float)
    parser.add_argument("--final_exploration", default=0.1, type=float)
    parser.add_argument("--final_exploration_frame", default=300000, type=int)
    parser.add_argument("--replay_start_size", default=50000, type=int)
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--no_render", action="store_true")

    args = parser.parse_args()

    qnet = DeepQNet(env_name=args.env, hidden_size=args.hidden_size)
    qnet.train(training_frames=args.training_frames,
               minibatch_size=args.minibatch_size,
               replay_memory_size=args.replay_memory_size,
               target_network_update_frequency=args.target_network_update_frequency,
               discount_factor=args.discount_factor,
               learning_rate=args.learning_rate,
               initial_exploration=args.initial_exploration,
               final_exploration=args.final_exploration,
               final_exploration_frame=args.final_exploration_frame,
               replay_start_size=args.replay_start_size,
               test=not args.no_test,
               render=not args.no_render)

    while True:
        qnet.play()
