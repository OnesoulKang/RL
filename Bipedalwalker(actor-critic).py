# import gym
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
n_rollout     = 10

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(24,256)
        #self.fc_pi = nn.Linear(256,8)

        self.fc_mu1 = nn.Linear(256,4)
        #self.fc_mu2 = nn.Linear(256,1)
        #self.fc_mu3 = nn.Linear(256,1)
        #self.fc_mu4 = nn.Linear(256,1)
        #self.fc_std1 = nn.Linear(256,1)
        #self.fc_std2 = nn.Linear(256,1)
        #self.fc_std3 = nn.Linear(256,1)
        #self.fc_std4 = nn.Linear(256,1)

        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim = 0, play=False):
        x = F.relu(self.fc1(x))

        mu1 = self.fc_mu1(x)

        if play:
            return mu1

        dist1 = Normal(mu1, 0.5)

        action1 = dist1.rsample()

        log_prob1 = dist1.log_prob(action1)

        real_action1 = torch.tanh(action1)

        return real_action1, log_prob1#, real_action2, real_action3, real_action4, log_prob1, log_prob2, log_prob3, log_prob4
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        #action1,action2,action3,action4,prob1,prob2,prob3,prob4 = self.pi(s)
        action1, _ = self.pi(s)
        dist = Normal(action1, 0.5)
        log_prob = dist.log_prob(a)
        #loss = -(torch.log(action1)+torch.log(action2)+torch.log(action3)+torch.log(action4)) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
        # loss = -torch.log(action1) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
        loss = -log_prob * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()         
      
def main():  
    env = gym.make('BipedalWalker-v3')
    test_env = gym.make('BipedalWalker-v3', render_mode="human")
    model = ActorCritic()    
    print_interval = 10
    score = 0.0
    epi_step = 0
    for n_epi in range(100000):

        if n_epi%200==0 and n_epi>0:
            for _ in range(1):
                # if test_env is None:
                #     test_env = gym.make("BipedalWalker-v3", render_mode="human")
                done = False
                step = 0
                s, _ = test_env.reset()
                while not done:
                    a = model.pi(torch.from_numpy(s).float(), play=True)
                    s_prime, r, done, _, _ = test_env.step(a.detach().numpy())
                    s = s_prime
                    score += r
                    step += 1
                    if done or step==1000:
                        break

        done = False
        s, _ = env.reset()

        while not done:
            for t in range(n_rollout):
                
                #a1, a2, a3, a4, p1, p2, p3, p4 = model.pi(torch.from_numpy(s).float())
                #
                # breakpoint()
                a, p = model.pi(torch.from_numpy(s).float())
                a = a.detach().numpy()

                #a = [a1.item(), a2.item(), a3.item(), a4.item()]
                
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data((s,a,r,s_prime,done))
                
                s = s_prime
                score += r
                epi_step += 1
                
                if done:
                    break                     
            
            model.train_net()
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, avg_step : {:.2f}, score/step : {:.2f}".format(n_epi, score/print_interval, epi_step/print_interval, score/epi_step))
            score = 0.0
            epi_step = 0
    env.close()

if __name__ == '__main__':
    main()