
  
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
class Memory():
    def __init__(self, max_capacity=2000):

        self.max_capacity = max_capacity
        self.log_probs = deque() #1
        self.rewards = deque() #2
        self.dones = deque() #3
        self.next_observations = deque() #4
        self.actions = deque() #5
        self.values= deque() #6

    def prepend_log_probs(self, val):
        if (len(self.log_probs)==self.max_capacity):
            self.log_probs.pop()
        self.log_probs.appendleft(val)

    def prepend_rewards(self, val):
        if (len(self.rewards)==self.max_capacity):
            self.rewards.pop()
        self.rewards.appendleft(val)

    def prepend_dones(self, val):
        if (len(self.dones)==self.max_capacity):
            self.dones.pop()
        self.dones.appendleft(val)

    def prepend_next_observations(self, val):
        if (len(self.next_observations)==self.max_capacity):
            self.next_observations.pop()
        self.next_observations.appendleft(val)

    def prepend_actions(self, val):
        if (len(self.actions)==self.max_capacity):
            self.actions.pop()
        self.actions.appendleft(val)
    
    def prepend_values(self, val):
        if (len(self.values)==self.max_capacity):
            self.values.pop()
        self.values.appendleft(val)
        

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


    


class Critic(nn.Module):
    def __init__(self,learning_rate):
        super().__init__()
        self.Layer_state_input = nn.Linear(in_features = 4, out_features=HIDDEN_SIZE )
        self.Layer_state_h1 = nn.Linear(in_features = HIDDEN_SIZE, out_features= HIDDEN_SIZE)
        self.Layer_state_h2 = nn.Linear(in_features = HIDDEN_SIZE, out_features=HIDDEN_SIZE )
        
        self.Layer_action_input = nn.Linear(in_features =1, out_features=HIDDEN_SIZE)
        self.Layer_action_h1 = nn.Linear(in_features = HIDDEN_SIZE, out_features=HIDDEN_SIZE )
        
        self.merged_input = nn.Linear(in_features= int(2*HIDDEN_SIZE), out_features =HIDDEN_SIZE)
        self.merged_h1 = nn.Linear(in_features= HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.merged_output =  nn.Linear(in_features= HIDDEN_SIZE, out_features=1)
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)

        
    def forward(self,obervations,action):
        

        t = F.relu(self.Layer_state_input(torch.tensor(obervations, dtype= torch.float32)))
        t = F.relu(self.Layer_state_h1(t))
        t = F.relu(self.Layer_state_h2(t))
         
        a = F.relu(self.Layer_action_input(torch.tensor([action], dtype= torch.float32)))
        a = F.relu(self.Layer_action_h1(a))
        
        merged = torch.cat((t,a), dim=0)
        merged = F.relu(self.merged_input(merged))
        merged = F.relu(self.merged_h1(merged))
        merged = self.merged_output(merged)
        return merged # returns future reward
        

class A2CAgent():
    def __init__(self, env, gamma = GAMMA, lr = LEARNING_RATE ):
        self.discount_rate = gamma
        self.learning_rate = lr
        self.env = env
        self.memory = Memory(MEM_CAPACITY)

        # Creating actor and critic networks
        self.actor = Actor(learning_rate= self.learning_rate)
        self.critic = Critic(learning_rate=self.learning_rate*10)
        self.train_cnt =0
    """
    actor takes in action and gives a 
    """
    
    def store(self, log_prob, reward, done, value):
        self.memory.prepend_log_probs(log_prob)
        self.memory.prepend_dones(done)
        self.memory.prepend_rewards(reward)
        self.memory.prepend_values(value)
    """
    In practice we would sample an action from the output of a network, apply this action in an environment, and then use log_prob to construct an equivalent loss function. Note that we use a negative because optimizers use gradient descent, whilst the rule above assumes gradient ascent. With a categorical policy, the code for implementing REINFORCE would be as follows:
    probs = policy_network(state)
    # Note that this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()
    """

    def play(self):
        episode_rewards = []
        for episode in range(TRAIN_EPISODES):
            avg = []
            observation = list(self.env.reset())
            total_reward = 0
            for t in range(500):
                log_action_probs = self.actor.forward(observation)
                action_probs = torch.exp(log_action_probs)
                distrubution = torch.distributions.Categorical(probs=action_probs)
                action = distrubution.sample()
                value = self.critic.forward(observation,action)
                next_observation, reward, done,  info =  env.step(int(action))
                
                self.store( distrubution.log_prob(action),reward,done,value)
                total_reward += reward
            
                #self.env.render()
                observation = next_observation
                
                if done:
                    self.train_cnt+=1
                    if total_reward >= 499:
                        self.save(self.actor, self.train_cnt)
                        
                    print(f"Training Episode finished ({self.train_cnt} of {TRAIN_EPISODES}).. Total Reward is : {total_reward}.")
                    last_q = self.critic.forward(next_observation, action)
                    self.train(t, torch.tensor([-100]))
                    episode_rewards.append(total_reward)
                    observation= self.env.reset()
                    break
        self.plot_graph(episode_rewards)

    def train_critic(self,t,q_val):
        # Critic
        assert self.memory.rewards[t]==1
        values = deque()
        for i in reversed(range(t)):
            values.appendleft(self.memory.values[i])
        values = torch.stack(list(values),1)
        q_values = deque()
        for i in reversed(range(t)):
            q_val = self.memory.rewards[i]+self.discount_rate * q_val  
            q_values.appendleft(q_val)
        self.q_values = torch.stack(list(q_values),1)
        # Instead of having the critic to learn the Q values, we make him learn the Advantage values.
        advantage = self.q_values - values
        loss = advantage.pow(2).mean()
        loss.backward()
        self.critic.optimizer.step() 
        self.critic.zero_grad()
            
           
    def train_actor(self,t):#update policy network
        #actor is trained for each time stamp
        assert self.memory.rewards[t]==1
        log_vals = []
        for i in range(t):
            log_vals.append(self.memory.log_probs[i])
        log_vals = torch.stack(log_vals,1)
        actor_loss = (-log_vals * self.q_values.detach()) *self.discount_rate 
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor.optimizer.step()
        self.actor.zero_grad()
    
    def train(self,t, last_q):
        self.train_critic(t, last_q)
        self.train_actor(t)

    def plot_graph(self,episode_rewards):
        plt.plot(np.arange(len(episode_rewards)), episode_rewards)#,s=2)
        plt.title("Total reward per episode (episodic)")
        plt.ylabel("reward")
        plt.xlabel("episode")
        plt.show()
    def save(self,primaryN,t):
        torch.save(primaryN.state_dict(), f"{t}-best.dat")

env = gym.make("CartPole-v1")
agent = A2CAgent(env)
agent.play()


episodes = 10
for episode in range(episodes):
    observation = env.reset()
    for t in range(501):
        action = agent.actor.forward(observation,probabilities=False )
        next_observation, reward, done,  info = env.step(int(action))
        env.render()

        
        if done:
            print(f"Game Episode finished in {t} timesteps.")
            break
