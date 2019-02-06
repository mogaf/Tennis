
import numpy as np
import random
import copy
from collections import namedtuple, deque

from ActorAgent import ActorAgent
from CriticAgent import CriticAgent
from noise import OUNoise
from replaybuff import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SharedCritic():

    def __init__(self,state_size,action_size,random_seed,num_agents):
        self.num_agents = num_agents
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.noise = OUNoise(action_size, random_seed)
        self.actors = [ActorAgent(i,state_size,action_size,random_seed,LR_ACTOR,self.noise,self.memory) for i in range(num_agents)]
        self.critic = CriticAgent(state_size, action_size, random_seed,LR_CRITIC,WEIGHT_DECAY,TAU)
        self.count = 0

    def act(self, states,add_noise=True):
        actions = []
        for actor, state in zip(self.actors, states):
            action = actor.act(state, add_noise=add_noise)
            actions.append(action)
        #return np.array(actions).reshape(1, -1) # reshape 2x2 into 1x4 dim vector
        return actions

    def reset(self):
        self.noise.reset()


    def step(self,states, actions, rewards, next_states, dones):
        for actor,state,action,reward,next_state,done in zip(self.actors,states, actions, rewards, next_states, dones):
            actor.step(state,action,reward,next_state,done)

        self.count = (self.count + 1) % UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE:
            if self.count == 0:
                for actor in self.actors:
                    experiences = self.memory.sample()
                    self.critic.learn(actor, experiences, GAMMA)
