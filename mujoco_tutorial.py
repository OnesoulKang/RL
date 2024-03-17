import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import argparse

#Hyperparameters
learning_rate  = 0.0003
gamma           = 0.99
lmbda           = 0.95
eps_clip        = 0.2
K_epoch         = 10
buffer_size    = 4
minibatch_size = 512
rollout_len    = buffer_size * minibatch_size

class PPO(nn.Module):
    def __init__(self, num_obs, num_act, use_std=False):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(num_obs,128)
        self.fc_mu = nn.Linear(128,num_act)
        
        self.use_std = use_std
        if self.use_std:
            self.fc_std  = nn.Linear(128,num_act)

        self.fc_v = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x))
        
        if self.use_std:
            std = F.softplus(self.fc_std(x))
        else:
            std = torch.tensor([0.5]) 

        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        for data in self.data:
            s, a, r, s_prime, prob_a, done = data
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        # s_batch.append(s_lst)
        # a_batch.append(a_lst)
        # r_batch.append(r_lst)
        # s_prime_batch.append(s_prime_lst)
        # prob_a_batch.append(prob_a_lst)
        # done_batch.append(done_lst)

        s_torch = torch.tensor(s_lst, dtype=torch.float, requires_grad=False)
        a_torch = torch.tensor(a_lst, dtype=torch.float, requires_grad=False)
        r_torch = torch.tensor(r_lst, dtype=torch.float, requires_grad=False)
        s_prime_torch = torch.tensor(s_prime_lst, dtype=torch.float, requires_grad=False)
        done_torch = torch.tensor(done_lst, dtype=torch.float, requires_grad=False)
        prob_a_torch = torch.tensor(prob_a_lst, dtype=torch.float, requires_grad=False)

        return s_torch, a_torch, r_torch, s_prime_torch, done_torch, prob_a_torch
    
    def calc_advantage(self, s, a, r, s_prime, done_mask):
        with torch.no_grad():
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
        delta = delta.numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)
        
        return td_target, advantage
        
    def train_net(self):
        if len(self.data) == minibatch_size * buffer_size:
            s, a, r, s_prime, done_mask, old_log_prob = self.make_batch()
            td_target, advantage = self.calc_advantage(s, a, r, s_prime, done_mask)

            idx = np.arange(minibatch_size * buffer_size)
            np.random.shuffle(idx)
            a_loss = []
            c_loss = []
            for i in range(K_epoch):
                for j in range(buffer_size):
                    id = idx[minibatch_size*j: minibatch_size*(j+1)]
                    s_batch, a_batch, r_batch, s_prime_batch, done_mask_batch, old_log_prob_batch, td_target_batch, advantage_batch = s[id], a[id], r[id], s_prime[id], done_mask[id], old_log_prob[id], td_target[id], advantage[id]

                    mu, std = self.pi(s_batch, softmax_dim=1)
                
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a_batch)
                
                    ratio = torch.exp(log_prob - old_log_prob_batch)  # a/b == exp(log(a)-log(b))
                
                    surr1 = ratio * advantage_batch
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage_batch
                
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s_batch) , td_target_batch)

                    a_loss.append(-torch.min(surr1, surr2).detach().mean().item())
                    c_loss.append(F.smooth_l1_loss(self.v(s_batch) , td_target_batch).detach().mean().item())

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1

            # print("{:^30}".format("==========AVG Loss=========="))
            # print("Avg. A_Loss {:.5f}".format(sum(a_loss)/len(a_loss)))
            # print("Avg. C_Loss {:.5f}".format(sum(c_loss)/len(c_loss)))
            self.data = []
        
def main(env):
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    # breakpoint()
    model = PPO(num_obs, num_act)
    score = 0.0
    print_interval = 20
    rollout = []

    s, _ = env.reset()
    done = False
    score = 0
 
    file_path = os.path.dirname(os.path.abspath(__file__))
    save_path = file_path+"/save/walker"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for n_epi in range(1000):
        for t in range(rollout_len):
            mu, std = model.pi(torch.from_numpy(s).float())
            dist = Normal(mu, std)

            a = dist.sample()
            log_prob = dist.log_prob(a)
            
            s_prime, r, done, truncated, info = env.step(a.numpy())
            
            # rollout.append((s, a.numpy(), r/10.0, s_prime, log_prob.detach().numpy() , done))
            # if len(rollout) == rollout_len:
            #     model.put_data(rollout)
            #     rollout = []
            model.put_data((s, a.numpy(), r/10.0, s_prime, log_prob.detach().numpy() , done))

            score += r

            if done:
                s, _ = env.reset()
                done = False
            else:
                s = s_prime

        model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, optmization step: {}".format(n_epi, score/print_interval, model.optimization_step))
            score = 0.0

            if n_epi % (print_interval * 5)==0 and n_epi != 0:
                play_env = gym.make("Walker2d-v4", render_mode="human")
                s_, _ = play_env.reset()
                done_ = False
                score_ = 0
                step_ = 0
                while not done_: #and step_<500:
                    mu_, _= model.pi(torch.from_numpy(s_).float())
                    s_p, r_, done_, _, _ = play_env.step(mu_.detach().numpy())
                    step_+=1
                    s_ = s_p
                    score_+=r_
                print("Play Result : {:.3f}".format(score_))

                torch.save({'model_state_dict':model.state_dict(),},
                           save_path+"/"+str(n_epi)+'.pt')
                
                play_env.close()
    env.close()


def play(env):
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    
    model = PPO(num_obs, num_act)

    file_path = os.path.dirname(os.path.abspath(__file__))
    save_path = file_path+"/save/walker"

    files = os.listdir(save_path)
    files.sort()
    
    load_file = save_path+"/"+files[-1]
    load_file = torch.load(load_file)

    model.load_state_dict(load_file['model_state_dict'])

    for n_epi in range(10):
        s, _ = env.reset()
        done = False
        score = 0.

        while not done:
            env.render()
            a, _ = model.pi(torch.from_numpy(s).float())
            
            s_prime, r, done, _, _ = env.step(a.detach().numpy())
            
            score += r

            s = s_prime

        print('{:<3} score : {:.3f}'.format('#'+str(n_epi), score))

    env.close()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Walker2d-v4')
    parser.add_argument('--play', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    
    if args.play:
        env = gym.make(args.env, render_mode="human")
        play(env)
    else:
        env = gym.make(args.env)
        main(env)