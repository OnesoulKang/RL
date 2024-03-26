import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
from network import ActorCritic
import time
from utils import *
import wandb

#Hyperparameters
learning_rate  = 0.0003
gamma           = 0.99
lmbda           = 0.95
eps_clip        = 0.2
K_epoch         = 10
buffer_size    = 4
minibatch_size = 512
rollout_len    = buffer_size * minibatch_size

class PPO():
    def __init__(self, network, save_path, 
                 imitation=False,
                 imitation_coef=1):
        super(PPO, self).__init__()
        self.data = []
        
        self.network = network
        self.save_path = save_path

        ##### Load Ref Net #####
        self.imitation = imitation
        if imitation:
            self.ref_network = ActorCritic(self.network.num_obs, self.network.num_actions)
            load(save_path+'/pre', self.ref_network)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.optimization_step = 0
        self.iter = 0
        self.imitation_coef = imitation_coef

    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
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

        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = np.array(s_prime_lst)
        done_lst = np.array(done_lst)
        prob_a_lst = np.array(prob_a_lst)

        s_torch = torch.tensor(s_lst, dtype=torch.float, requires_grad=False)
        a_torch = torch.tensor(a_lst, dtype=torch.float, requires_grad=False)
        r_torch = torch.tensor(r_lst, dtype=torch.float, requires_grad=False)
        s_prime_torch = torch.tensor(s_prime_lst, dtype=torch.float, requires_grad=False)
        done_torch = torch.tensor(done_lst, dtype=torch.float, requires_grad=False)
        prob_a_torch = torch.tensor(prob_a_lst, dtype=torch.float, requires_grad=False)

        return s_torch, a_torch, r_torch, s_prime_torch, done_torch, prob_a_torch
    
    def calc_advantage(self, s, a, r, s_prime, done_mask):
        with torch.no_grad():
            td_target = r + gamma * self.network.v(s_prime) * done_mask
            delta = td_target - self.network.v(s)
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
            i_loss = []
            for i in range(K_epoch):
                for j in range(buffer_size):
                    id = idx[minibatch_size*j: minibatch_size*(j+1)]
                    s_batch, a_batch, r_batch, s_prime_batch, done_mask_batch, old_log_prob_batch, td_target_batch, advantage_batch = s[id], a[id], r[id], s_prime[id], done_mask[id], old_log_prob[id], td_target[id], advantage[id]

                    mu, std = self.network.pi(s_batch, softmax_dim=1)
                
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a_batch)
                
                    ratio = torch.exp(log_prob - old_log_prob_batch)  # a/b == exp(log(a)-log(b))
                
                    surr1 = ratio * advantage_batch
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage_batch

                    if self.imitation:
                        ref_actions, _ = self.ref_network.pi(s_batch, softmax_dim=1)
                        imitation_loss = F.smooth_l1_loss(mu, ref_actions.detach())
                        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.network.v(s_batch) , td_target_batch) + self.imitation_coef * imitation_loss.mean()
                        i_loss.append(imitation_loss.mean().detach().item())
                    else:
                        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.network.v(s_batch) , td_target_batch)

                    a_loss.append(-torch.min(surr1, surr2).detach().mean().item())
                    c_loss.append(F.smooth_l1_loss(self.network.v(s_batch) , td_target_batch).detach().mean().item())

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1

            self.data = []
            self.iter += 1

            if self.iter % 100 == 0:
                torch.save({'model_state_dict':self.network.state_dict(),},
                            self.save_path+"/"+str(self.iter)+'.pt')
            
            infos = {'a_loss': sum(a_loss)/len(a_loss),
                     'c_loss': sum(c_loss)/len(c_loss),}
            if self.imitation:
                infos['i_loss'] = sum(i_loss)/len(i_loss)

            return infos
        
def main(env, args):
    file_path = os.path.dirname(os.path.abspath(__file__))
    save_path = file_path+"/save/walker"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    ppo_args = {'imitation': args.imitation,
                'imitation_coef': float(args.imitation_coef)}
    net = ActorCritic(num_obs, num_act)
    alg = PPO(net, save_path, **ppo_args)
    
    score = 0.0
    print_interval = 20

    s, _ = env.reset(seed=args.seed)
    done = False
    score = 0
    episode_scores = []
 
    times = []
    max_iter = 1000

    wandb.init(project='Imitation Test')
    # wandb.run.name()
    wandb.run.save()
    for n_epi in range(max_iter):
        start = time.time()
        for t in range(rollout_len):
            mu, std = alg.network.pi(torch.from_numpy(s).float())
            dist = Normal(mu, std)

            a = dist.sample()
            log_prob = dist.log_prob(a)
            
            s_prime, r, done, truncated, info = env.step(a.numpy())
            
            alg.put_data((s, a.numpy(), r/10.0, s_prime, log_prob.detach().numpy() , done))

            score += r

            if done:
                s, _ = env.reset()
                done = False
                episode_scores.append(score)
                score = 0
            else:
                s = s_prime

        infos = alg.train_net()
        end = time.time()
        times.append(end-start)
        logs = {'loss/a_loss': infos['a_loss'],
                'loss/c_loss': infos['c_loss'],}
        
        if len(episode_scores) != 0:
            logs['rewards']= sum(episode_scores)/len(episode_scores)
            episode_scores.clear()

        if 'i_loss' in infos:
            logs['loss/i_loss'] = infos['i_loss']

        wandb.log(logs)

        if n_epi%print_interval==0 and n_epi!=0:
            tmp = "Iterations {}/{} ({} steps/sec)".format(n_epi, max_iter, int(rollout_len*print_interval/sum(times)))
            print("\n{:^100}\navg score : {:.1f}, optmization step: {}".format(tmp, score/print_interval, alg.optimization_step))
            print("Avg. A_Loss {:.5f}".format(infos['a_loss']))
            print("Avg. C_Loss {:.5f}".format(infos['c_loss']))
            if 'i_loss' in infos:
                print("Avg. I_Loss {:.5f}".format(infos['i_loss']))

            times.clear()

            # if n_epi % (print_interval * 5)==0 and n_epi != 0:
            #     play_env = gym.make("Walker2d-v4", render_mode="human")
            #     s_, _ = play_env.reset()
            #     done_ = False
            #     score_ = 0
            #     step_ = 0
            #     while not done_: #and step_<500:
            #         mu_, _= alg.network.pi(torch.from_numpy(s_).float())
            #         s_p, r_, done_, _, _ = play_env.step(mu_.detach().numpy())
            #         step_+=1
            #         s_ = s_p
            #         score_+=r_
            #     print("Play Result : {:.3f}".format(score_))
                
            #     play_env.close()
    env.close()


def play(env):
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    
    net = ActorCritic(num_obs, num_act)

    file_path = os.path.dirname(os.path.abspath(__file__))
    save_path = file_path+"/save/walker"

    load(save_path, net)
    
    for n_epi in range(10):
        s, _ = env.reset()
        done = False
        score = 0.

        while not done:
            env.render()
            a, _ = net.pi(torch.from_numpy(s).float())
            
            s_prime, r, done, _, _ = env.step(a.detach().numpy())
            
            score += r

            s = s_prime

        print('{:<3} score : {:.3f}'.format('#'+str(n_epi), score))

    env.close()

if __name__ == '__main__':
    args = get_arguments()
    set_seed(args.seed)

    if args.play:
        env = gym.make(args.env, render_mode="human")
        play(env)
    else:
        env = gym.make(args.env)
        main(env, args)