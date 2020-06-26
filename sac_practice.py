import math
import random
import os
import time
import argparse

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--play', default=False, action="store_true")
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--load_trained_model', type=int, default=None)
parser.add_argument('--control_time', type=float, default=0.01)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

current_path = os.path.dirname(os.path.abspath(__file__))

if not os.path.isdir(current_path+"/logdir/SAC"):
    os.makedirs(current_path+'/logdir/SAC')

if not os.path.isdir(current_path+"/saved_model/SAC"):
    os.makedirs(current_path+'/saved_model/SAC')    

model_path = current_path+'/saved_model/SAC'
writer = SummaryWriter(current_path+'/logdir/SAC')

gamma = 0.99
lr = 0.0003
alpha = 0.01
tau = 0.005

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer=[]
        self.idx = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = (state, action, reward, next_state, done)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=1e-3):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        state = torch.cuda.FloatTensor(state)
        action = torch.cuda.FloatTensor(action)
        x = torch.cat([state, action], 1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=1e-3, log_std_min=-5, log_std_max=0.01):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        #self.mean_linear.weight.data.uniform_(-init_w, init_w)
        #self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        #self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        #self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        #mean = torch.relu(self.mean_linear(x))
        mean = self.mean_linear(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def get_action(self, state):
        state = torch.cuda.FloatTensor(state)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        #x_t = normal.rsample()
        dist = Normal(0,1)
        e = dist.sample().to(device)
        u = mean + e * std
        y_t = torch.tanh(u)
        
        logp = normal.log_prob(u).sum(dim=-1, keepdim=True) - torch.log(1 - y_t ** 2 + 1e-6).sum(dim=-1, keepdim=True)                
        return y_t.cpu().data.numpy(), logp

def hard_update(target, origin):
    for target_param, param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(param.data)

def soft_update(target, origin, tau):
    for target_param, param in zip(target.parameters(), origin.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def update(data, q1, q1_optim, q2, q2_optim, q1_target, q2_target, policy, policy_optim, total_step):
    state, action, next_state, reward, done = data

    d = done.astype(int)
    d = torch.cuda.FloatTensor(d)

    q1_value = q1(state, action)
    q2_value = q2(state, action)
    
    with torch.no_grad():
        a2, logp_a2 = policy.get_action(next_state)

        q1_target_value = q1_target(next_state, a2)
        q2_target_value = q2_target(next_state, a2)
        min_q_target = torch.min(q1_target_value, q2_target_value)

        backup = torch.cuda.FloatTensor(reward) + gamma * (1-d) * (min_q_target - alpha * logp_a2)

    loss_q1 = ((q1_value - backup)**2).mean()
    loss_q2 = ((q2_value - backup)**2).mean()
    
    writer.add_scalar('loss/q1', loss_q1, total_step)
    writer.add_scalar('loss/q2', loss_q2, total_step)
    
    q1_optim.zero_grad()    
    loss_q1.backward()
    q1_optim.step()

    q2_optim.zero_grad()
    loss_q2.backward()
    q2_optim.step()

    soft_update(q1_target, q1, tau)
    soft_update(q2_target, q2, tau)

    #### Q update End, Policy update Start

    pi, logp_pi = policy.get_action(state)    
    q1_pi = q1(state, pi)
    q2_pi = q2(state, pi)
    q_pi = torch.min(q1_pi, q2_pi)

    loss_pi = (alpha * logp_pi - q_pi).mean()
    writer.add_scalar('loss/pi', loss_pi, total_step)

    policy_optim.zero_grad()
    loss_pi.backward()
    policy_optim.step()    

def main():
    #env = gym.make('Aidinvi_standing-v0', is_render = args.render, )    
    env = gym.make('Pendulum-v0')
    
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    hidden_dim = 256
    
    # Q_function and Target_Q_function
    q1, q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device), QNetwork(state_dim, action_dim, hidden_dim).to(device)
    q1_optim, q2_optim = optim.Adam(q1.parameters(), lr), optim.Adam(q2.parameters(), lr)
    q1_target, q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device), QNetwork(state_dim, action_dim, hidden_dim).to(device)

    hard_update(q1_target, q1)
    hard_update(q2_target, q2)

    for param in q1_target.parameters():
        param.requires_grad = False

    for param in q2_target.parameters():
        param.requires_grad = False

    policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_optim = optim.Adam(policy.parameters(), lr)

    total_step = 0      
    max_episode = 10000
    minimum_buffer_size=5000
    update_every = 2
    batch_size = 256
    
    buffer_size = 100000
    buffer = ReplayBuffer(buffer_size)

    for episode_number in range(max_episode):
        state, step, episode_reward, done = np.array(env.reset()), 0, 0, False
        state = torch.cuda.FloatTensor(state)

        while not done:
            if total_step < minimum_buffer_size:
                action = env.action_space.sample()
            else:
                action, _ = policy.get_action(state)

            next_state, reward, done, _ = env.step(action)
            
            buffer.push(state.cpu().data.numpy(), action, next_state, reward, done)
            state = torch.FloatTensor(next_state).to(device)
            episode_reward += reward
            step += 1        
            total_step +=1

            if (total_step >= minimum_buffer_size and total_step%update_every == 0):                
                batch = buffer.sample(batch_size)
                update(batch, q1, q1_optim, q2, q2_optim, q1_target, q2_target, policy, policy_optim, total_step)
        
        print('#EP {0} | episode_step {1} | reward {2} | total_step {3}'.format(episode_number, step, episode_reward, total_step))
        
        if total_step>minimum_buffer_size:
            writer.add_scalar('episode_reward', episode_reward, episode_number)     

            if episode_number%100==0:
                model_save = os.path.join(model_path, 'ckpt_epi_'+str(episode_number)+'_pth.tar')    
                torch.save({
                            'actor_state_dict':policy.state_dict(),
                            #'critic_state_dict':critic.state_dict(),
                            #'actor_optimizer_state_dict':actor.optimizer.state_dict(),
                            #'critic_optimizer_state_dict':critic.optimizer.state_dict(),
                }, model_save)                

def play():
    pass
    '''
    if args.load_trained_model == None:
        print("* * * * * Enter a model score * * * * *")
        quit()
    elif not os.path.isfile(model_path + '/ckpt_'+str(args.load_trained_model)+'_pth.tar'):
        print("* * * * * File not exist * * * * *")
        quit()
    
    env = gym.make('Aidinvi_standing-v0', is_render = True, )
    
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0] + 6
    hidden_dim = 256

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
    load_model = '/ckpt_'+str(args.load_trained_model)+'_pth.tar'
    checkpoint = torch.load(model_path+load_model)

    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    control_time = args.control_time
    while True:
        state = env.reset(epoch=0, play=True)
        start_time=time.time()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            step_start = time.time()
            action = policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action)
            step += 1
            state = next_state.tolist()
            episode_reward += reward
            
            time_to_sleep = control_time-(time.time()-step_start)
            if time_to_sleep>0:
                time.sleep(time_to_sleep)            

            if done:                  
                print('step : {0} | score : {1} | time : {2} s'.format(step, episode_reward, time.time()-start_time))            
        '''

if __name__ == '__main__':
    torch.manual_seed(0)
    if not args.play:
        main()
    else:
        play()