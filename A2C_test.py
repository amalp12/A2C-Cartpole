  
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from collections import deque
import random
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("error")

GAMMA = 0.9
LEARNING_RATE = 0.0009
MEM_CAPACITY = 520
TRAIN_EPISODES = 10000
HIDDEN_SIZE = 64
class Actor(nn.Module):
    def __init__(self,learning_rate):
        super().__init__()
        self.Layer_1 = nn.Linear(in_features = 4, out_features=HIDDEN_SIZE )
        self.Layer_2 = nn.Linear(in_features = HIDDEN_SIZE, out_features=HIDDEN_SIZE )
        self.Layer_3 = nn.Linear(in_features = HIDDEN_SIZE, out_features=HIDDEN_SIZE )
        self.Layer_4 = nn.Linear(in_features = HIDDEN_SIZE, out_features=2 )
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)


    def forward(self, observations, probabilities = True):
        assert type(observations) != torch.Tensor
        t = F.relu(self.Layer_1(torch.tensor([observations], dtype =torch.float32)))
        t = F.relu(self.Layer_2(t))
        t = F.relu(self.Layer_3(t))
        t = F.log_softmax(self.Layer_4(t), dim=1)
        if probabilities:
            prediction = t
        else :
            prediction = torch.argmax(t)
        return prediction


    

    

env = gym.make('CartPole-v1')
#env = env.unwrapped

targetNet = Actor(0.01)

#targetNet.load_state_dict(torch.load("92-best.dat"))
#targetNet.load_state_dict(torch.load("100-best.dat"))
#targetNet.load_state_dict(torch.load("100-best.dat"))
targetNet.load_state_dict(torch.load("A2C_best.dat"))



avg =[]
for i_episode in range(20):
    
    observation = env.reset()
    for t in range(500):
        env.render()
        #print(observation)
        # Action is to be chosen by the agent
        #action = env.action_space.sample()
        action = targetNet.forward(list(observation),probabilities=False)
        observation, reward, done, info = env.step(int(action))
        #print("Not Done!")
        if done:
            avg.append(t+1)
            print("Game Episode finished after {} timesteps".format(t+1))
            break

print("Average Score: ",sum(avg)/len(avg))
env.close()


