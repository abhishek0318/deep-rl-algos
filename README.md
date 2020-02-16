# Deep Reinforcement Learning Algorithms

This repository contains my implementations of some of the popular Deep Reinforcement Learning algorithms. I have focussed on code readability and tried to keep each algorithm's code as self contained as possible. So there might be some code repetition and inefficiencies.   

## Algorithms

- [x] Vanilla Policy Gradient (VPG) - [Code](./vpg.py), [Pseudo Code](https://spinningup.openai.com/en/latest/_images/math/262538f3077a7be8ce89066abbab523575132996.svg)
- [x] Deep Q Network (DQN) - [Code](https://github.com/abhishek0318/dqn-atari/blob/master/dqn.py), [Pseudo Code](https://miro.medium.com/max/1206/1*nb61CxDTTAWR1EJnbCl1cA.png) 
- [x] Asynchronous Advantage Actor Critic (A3C) [Code](./a3c.py) [Pseudo Code](http://shaofanlai.com/archive/storage/trZeYicStx3WUt1BZwAvKY7g2K7kn8zGrvOYEAk5QjQEUDzotX)
- [ ] Proximal Policy Optimization (PPO)
- [ ] Deep Deterministic Policy Gradients (DDPG)

## Requirements

This code is written for Python 3.6 and PyTorch 1.1. Install the dependencies by the following command,

> pip install -r requirements.txt