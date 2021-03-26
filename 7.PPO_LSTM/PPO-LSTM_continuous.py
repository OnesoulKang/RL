#PPO-LSTM
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchsummary import summary as summary_
import time
import numpy as np
import matplotlib.pyplot as plt
import os

#Hyperparameters
learning_rate = 0.0003
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 5
T_horizon     = 20
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
file_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(file_path, 'saved_model')

class PPO(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PPO, self).__init__()
        self.data = []
        self.action_dim = action_dim
        
        self.fc1   = nn.Linear(obs_dim,128)
        self.lstm  = nn.LSTM(128,64) # input_size = 128, hidden_size=64, batch_first=False
        self.fc_pi = nn.Linear(64,action_dim)
        self.fc_v  = nn.Linear(64,1)

        self.log_std = np.ones(action_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.Tensor(self.log_std).to(dev))
        self.log_std = torch.Tensor(self.log_std).to(dev)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden, use_pi = True, a = None):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 128) # seq_len, batch_size, feature_size
        x, lstm_hidden = self.lstm(x, hidden)
        x = torch.tanh(self.fc_pi(x))

        # sigma = torch.FloatTensor([0]).expand_as(x).to(dev)

        p_action = torch.distributions.Normal(2 * x, self.log_std.exp())
        action = p_action.sample()

        if use_pi:
            prob = p_action.log_prob(action)
        else:
            a = a.unsqueeze(2)
            prob = p_action.log_prob(a)
            return prob.squeeze(1)

        #prob = F.softmax(x, dim=2)
        return action.cpu().data.numpy(), lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 128)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask = torch.tensor(s_lst, dtype=torch.float).to(dev), torch.tensor(a_lst, dtype=torch.float).to(dev), \
                                         torch.tensor(r_lst, dtype=torch.float).to(dev), torch.tensor(s_prime_lst, dtype=torch.float).to(dev), \
                                         torch.tensor(done_lst, dtype=torch.float).to(dev)
        self.data = []
        return s,a,r,s_prime, done_mask, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()        
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        # log_prob_a_old(s,a)
        prob_a = self.pi(s, first_hidden, use_pi=False, a = a)
        prob_a = prob_a.detach()

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().cpu().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(dev)

            # log_prob_a_new(s,a)
            pi = self.pi(s, first_hidden, use_pi=False, a = a)            
            ratio = torch.exp(pi - prob_a)  # a/b == log(exp(a)-exp(b))
            assert pi.size() == prob_a.size(), f'pi {pi.size()}, prob_a {prob_a.size()}'

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            assert ratio.size() == advantage.size(), '{}, {}'.format(ratio.size(), advantage.size())

            loss1 = -torch.min(surr1, surr2).mean()
            loss2 = F.smooth_l1_loss(v_s, td_target.detach())
            loss = loss1 + 0.5 * loss2

            assert loss1.size() == loss2.size(), f'loss1 {loss1.size()}, loss2 {loss2.size()}'

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
        
def main():
    env = gym.make('Pendulum-v0')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = PPO(obs_dim, action_dim).to(dev)
    score = 0.0
    print_interval = 10
    reward_record = [[],[]]
    
    for n_epi in range(5000):
        h_out = (torch.zeros([1, 1, 64], dtype=torch.float, device=dev), torch.zeros([1, 1, 64], dtype=torch.float, device=dev)) # h_0, c_0 for LSTM
        s = env.reset()
        done = False
        model.eval()
        
        while not done:
            # for t in range(T_horizon):
            # env.render()
            h_in = h_out
            a, h_out = model.pi(torch.from_numpy(s).float().to(dev), h_in)
            #a = np.array([a.item()])
            a = a.item()
            # a = np.clip(a, -2, 2)
            # m = Categorical(prob)
            # a = m.sample().item()
            s_prime, r, done, info = env.step(np.array([a]))

            model.put_data((s, a, r, s_prime, h_in, h_out, done))
            s = s_prime

            score += r
            if done:
                break
                    
        model.train()
        model.train_net()
        model.eval()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            reward_record[0].append(n_epi)
            reward_record[1].append(score/print_interval)
            score = 0.0

        if n_epi%200==0 and n_epi!=0:
            model_save = os.path.join(path, 'ckpt_'+str(int(n_epi))+'_pth.tar')    
            torch.save({
                        'model_state_dict':model.state_dict(),                        
                        'model_optimizer_state_dict':model.optimizer.state_dict(),                        
            }, model_save) 

    h_out = (torch.zeros([1, 1, 64], dtype=torch.float, device=dev), torch.zeros([1, 1, 64], dtype=torch.float, device=dev)) # h_0, c_0 for LSTM
    s = env.reset()
    done = False
    score = 0

    while not done:                
        env.render()
        h_in = h_out
        a, h_out = model.pi(torch.from_numpy(s).float().to(dev), h_in)
        a = a.item() * 2        
        # prob = prob.view(-1)
        # m = Categorical(prob)
        # a = m.sample().item()
        s_prime, r, done, info = env.step(np.array([a]))
        s = s_prime

        score += r
        if done:
            print(f'=====evaluation reward : {score}=====')        
            break

    env.close()

    plt.plot(reward_record[0], reward_record[1])
    plt.show()


if __name__ == '__main__':
    main()