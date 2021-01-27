import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import os
import numpy as np
import argparse
import time
import pybullet_envs

# from scipy import fftpack
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--train', default=False, action="store_true")
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--tensorboard', default=False, action="store_true")
parser.add_argument('--T-horizon', type=int, default=200)
parser.add_argument('--max-eps', type=int, default=10000)
parser.add_argument('--load_model_score', type=int, default=1035)
parser.add_argument('--load_trained_model', type=int, default=None)
parser.add_argument('--mini-batch-size', type=int, default=64)
parser.add_argument('--NP', type=int, default=1)
parser.add_argument('--control_time', type=float, default=0.001)
parser.add_argument('--ENV', type=str, default='HopperBulletEnv-v0')

torch.autograd.set_detect_anomaly(True)

args = parser.parse_args()

ENV_NAME = args.ENV
learning_rate = 0.0001
entropy_coef  = 0.0
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.15
K_epoch       = 10 #5
T_horizon     = args.T_horizon
render = args.render

#####path control#####
current_path = os.path.dirname(os.path.abspath(__file__))
save_path = "/sigma_-1_lr_"+str(learning_rate)+"_ent_coef_"+str(entropy_coef)+"_hopper"
if not os.path.isdir(current_path+"/logdir"+save_path):
    os.makedirs(current_path+'/logdir'+save_path)

if not os.path.isdir(current_path+"/saved_model"+save_path):
    os.makedirs(current_path+'/saved_model'+save_path)    

model_path = current_path+'/saved_model'+save_path

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# sigma = torch.cuda.FloatTensor([0.05]* 12)

class Actor(nn.Module):
    def __init__(self, n_in,n_mid, n_out, log_std_min=-4, log_std_max=-1):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

        self.sigma = nn.Linear(n_mid, n_out)

        self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.fc3.bias.data.uniform_(-0.003, 0.003)
        # self.sigma.weight.data.uniform_(-0.003, 0.003)
        # self.sigma.bias.data.uniform_(-0.003, 0.003)
        
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
    
    def forward(self, x, play=False):
        #print(self.fc1.weight)
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        mean_output = torch.tanh(self.fc3(h2))
        sigma_output = torch.tanh(self.sigma(h2))        

        sigma = torch.cuda.FloatTensor([-1]).expand_as(mean_output)
        #sigma = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (sigma_output + 1)
        
        if play==True:
            return 0, mean_output.cpu().data.numpy(), sigma

        sampled_actions = self.get_action(mean_output, sigma) 

        sampled_actions = sampled_actions.cpu()
        sampled_actions = sampled_actions.data.numpy()
        sampled_actions = np.clip(sampled_actions, -1, 1)
        return mean_output, sampled_actions, sigma

    def get_action(self, mean_output, sigma):    
        actions = torch.distributions.Normal(mean_output, sigma.exp()) # exp(-2) = 0.13534, exp(-4) = 0.018316
        sampled_actions = actions.sample()        
        return sampled_actions

    def get_prob(self, s, a):
        mean_output, _, sigma = self.forward(s)
        actions = torch.distributions.Normal(mean_output,sigma.exp())
        log_probs = actions.log_prob(a)
        #log_probs = torch.log(torch.exp(-(a-mean_output)**2/(2*sigma.exp())/torch.sqrt(2*torch.cuda.FloatTensor([3.14])*sigma.exp())))
        entropy = actions.entropy() # 0.5 * ln(2*pi*e) + ln(std.dev.)
        entropy = entropy.sum(-1).mean()
        return log_probs, entropy       


class Critic(nn.Module):
    def __init__(self, n_in,n_mid):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = 3*learning_rate)
    
    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        v = self.fc3(h2)
        
        return v

def train_model(actor, critic, memory, writer, epoch):
    s_lst, a_lst, r_lst, s_prime_lst, done_lst = [],[],[],[],[]
    mini_batch_size = args.mini_batch_size
    for transition in memory:
        s, a, r, s_prime, done = transition

        s_lst.append(s)
        a_lst.append(a)
        r_lst.append(r)
        s_prime_lst.append(s_prime)
        done_mask = 0 if done else 1
        done_lst.append([done_mask])

    s, a, r, s_prime, done = torch.cuda.FloatTensor(s_lst), torch.cuda.FloatTensor(a_lst), \
                                     torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                     torch.tensor(done_lst)
    n = len(s)
    arr = np.arange(n)

    old_values = critic(s).detach()
    old_policy, _ = actor.get_prob(s, a) #.detach()
    old_policy = old_policy.detach()
    
    #####
    returns = torch.zeros_like(r)
    advantage = torch.zeros_like(r)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, n)):
        running_returns = r[t] + gamma * running_returns * done[t]
        running_tderror = r[t] + gamma * previous_value * done[t] - old_values.cpu().data[t]
        running_advants = running_tderror + gamma * lmbda * running_advants * done[t]

        returns[t] = running_returns
        previous_value = old_values.cpu().data[t]
        advantage[t] = running_advants

    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6) # 정규화

    criterion = torch.nn.MSELoss()    

    for k in range(K_epoch):
        np.random.shuffle(arr)

        for i in range(n // mini_batch_size):
            batch_index = arr[mini_batch_size * i: mini_batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = s[batch_index]
            returns_samples = returns.to(dev).detach()[batch_index]
            advants_samples = advantage[batch_index]
            actions_samples = a[batch_index]
            old_policy_samples = old_policy[batch_index]
            #target_samples = td_target.detach()[batch_index]

            ####
            log_probs, entropy = actor.get_prob(inputs, actions_samples)
            ratio = torch.exp(log_probs - old_policy_samples) 
            ####

            surr1 = ratio * advants_samples.to(dev)
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advants_samples.to(dev)
            actor_loss = -(torch.min(surr1, surr2).mean() + entropy_coef * entropy) # entropy sign +? -?

            new_value = critic(inputs)
            critic_loss = criterion(new_value, returns_samples) #+ entropy_coef * entropy

            critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            if not torch.isfinite(critic_loss):
                print('WARNING: non-finite critic_loss, ending training ')
                exit(1)
            # torch.nn.utils.clip_grad_norm_(critic.parameters(),
            #                 0.05
            #                 )
            critic.optimizer.step()

            actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            if not torch.isfinite(actor_loss):
                print('WARNING: non-finite actor_loss, ending training ')
                print(actor_loss)
                exit(1)
            # torch.nn.utils.clip_grad_norm_(actor.parameters(),
            #                 0.05
            #                 )
            actor.optimizer.step()
            
            if k==(K_epoch-1):
                if args.tensorboard:
                    writer.add_scalar('loss/actor', actor_loss, epoch)
                    writer.add_scalar('loss/critic', critic_loss, epoch)
                    #writer.add_scalar('loss/entropy', entropy, epoch)

def data_func(env, actor, critic, train_queue, coordinator, best_score, idx):
    if args.tensorboard:
        tensorboard_path = current_path+'/logdir'+save_path+'/Process-'+str(idx)
        writer = SummaryWriter(tensorboard_path)

    env = gym.make(ENV_NAME)
    actor.eval()
    step = 0
    score = 0.0
    total_step = 0
    episode_num = 0
    memory = []
    max_reward = 2.0 * 200

    #filtered_action = np.zeros(12)
    #alpha = 0.6141 # cut_off 5hz

    for epoch in range(args.max_eps):
        while True:
            s = np.array(env.reset())
            done = False
            while not done:            
                if idx == 0:
                    sampled_actions, sigma = actor.forward(torch.cuda.FloatTensor(s))[1:3]
                else:
                    sampled_actions = actor.forward(torch.cuda.FloatTensor(s))[1]
                
                if args.tensorboard and idx==0:
                    ent = actor.get_prob(torch.cuda.FloatTensor(s), torch.cuda.FloatTensor(sampled_actions))[1]
                    writer.add_scalar('loss/entropy', ent, total_step+step)                
                    writer.add_scalar('value/sigma', sigma[0], total_step+step)

                s_prime, r, done, _ = env.step(sampled_actions)
                
                memory.append((s, sampled_actions, [r], s_prime, done)) ## S_t, a_t, r_t, S_t+1, d_t+1

                step += 1            
                score += r
                s = s_prime
            
            episode_num +=1
            
            if step > T_horizon:
                break
        
        train_queue.put(memory)
        memory = []

        avg_step = step / episode_num
        avg_score = score / episode_num
        
        if args.tensorboard:
            writer.add_scalar('value/reward', avg_score, epoch)
            writer.add_scalar('value/normalized_reward', avg_score/max_reward, epoch)
            writer.add_scalar('value/episode_step', avg_step, epoch)
            writer.add_scalar('value/reward_per_step', avg_score/avg_step, epoch)            

        total_step += step
        step = 0
        score = 0.0
        episode_num = 0

        print('Process-{0} | epoch : {3} | avg_step : {1} | avg_score : {2}'.format(idx, avg_step, avg_score, epoch))

        if avg_score>best_score.value:
            model_save = os.path.join(model_path, 'ckpt_'+str(int(avg_score))+'_pth.tar')    
            torch.save({
                        'actor_state_dict':actor.state_dict(),
                        'critic_state_dict':critic.state_dict(),
                        'actor_optimizer_state_dict':actor.optimizer.state_dict(),
                        'critic_optimizer_state_dict':critic.optimizer.state_dict(),
            }, model_save)
            print('**best score update** {0} --> {1}'.format(best_score.value, avg_score))
            best_score.value = avg_score
        
        coordinator[idx]= 0

        while not coordinator[idx]:
            pass
    
    print('Process-{0} finished. Total {1} step'.format(idx, total_step))
    env.close()    

def train():
    #####load model#####
    if (not args.load_model_score == None) or (not args.load_trained_model == None):
        if not args.load_model_score == None:
            load_model = '/ckpt_'+str(args.load_model_score)+'_pth.tar'
        else:
            load_model = '/ckpt_train_'+str(args.load_trained_model)+'_pth.tar'
        
        if not os.path.isfile(model_path+load_model):
            print('!!!File is not exist!!!')
            quit()
            
    mp.set_start_method('spawn')

    env = gym.make(ENV_NAME)

    action_size = env.action_space.shape[0]
    observation_size = env.observation_space.shape[0]
    hidden_size = 256

    #env.close()

    actor = Actor(observation_size, hidden_size, action_size).to(dev)
    critic = Critic(observation_size, hidden_size).to(dev)

    if args.tensorboard:
        tensorboard_path = current_path+'/logdir'+save_path+'/Main_Process'
        writer = SummaryWriter(tensorboard_path)

    if (not args.load_model_score == None) or (not args.load_trained_model == None):
        checkpoint = torch.load(model_path+load_model)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        critic.optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])    

    PROCESS_NUMBER = args.NP
    train_queue = mp.Queue()
    coordinator = mp.Array('i', [1] * PROCESS_NUMBER)
    best_score = mp.Value('f', -1000)
    coordinator_np = np.array([1] * PROCESS_NUMBER)

    data_proc_list = []
    batch = []
    num_train = 0

    #####spawn processes#####
    for i in range(PROCESS_NUMBER):
        data_proc = mp.Process(target=data_func, args=(env, actor, critic, train_queue, coordinator, best_score, i))
        data_proc.start()
        data_proc_list.append(data_proc)

    while True:
        for i in range(PROCESS_NUMBER):
            coordinator_np[i] = coordinator[i]

        if not train_queue.empty():
            train_entry = train_queue.get()
            batch.extend(train_entry)
        
        if len(batch)<T_horizon*PROCESS_NUMBER:
            continue
        
        if np.all(coordinator_np==0):
            num_train +=1
            print('update_model start')
            train_model(actor, critic, batch, writer, num_train)
            print('update_model end')
            
            batch = []
            
            if num_train%50==0:
                model_save = os.path.join(model_path, 'ckpt_train_'+str(num_train)+'_pth.tar')    
                torch.save({
                        'actor_state_dict':actor.state_dict(),
                        'critic_state_dict':critic.state_dict(),
                        'actor_optimizer_state_dict':actor.optimizer.state_dict(),
                        'critic_optimizer_state_dict':critic.optimizer.state_dict(),
                }, model_save)

            for i in range(PROCESS_NUMBER):
                coordinator[i] = 1
        
        if num_train>args.max_eps:
            break

    for p in data_proc_list:
        p.terminate()
        p.join()

def play():
    if (not args.load_model_score == None) or (not args.load_trained_model == None):
        if not args.load_model_score == None:
            load_model = '/ckpt_'+str(args.load_model_score)+'_pth.tar'
        else:
            load_model = '/ckpt_train_'+str(args.load_trained_model)+'_pth.tar'
        
        if not os.path.isfile(model_path+load_model):
            print('!!!File is not exist!!!')
            quit()

    env = gym.make(ENV_NAME, render=True)

    action_size = env.action_space.shape[0]
    observation_size = env.observation_space.shape[0] 
    hidden_size = 256

    actor = Actor(observation_size, hidden_size, action_size).to(dev)
    critic = Critic(observation_size, hidden_size).to(dev)
    
    control_time = args.control_time
    #####load model#####        
    checkpoint = torch.load(model_path+load_model)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    critic.optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])    

    step = 0
    score = 0.0

    print("1 episode or 1 step? insert 'e' or 's'")
    user_input = user_key_input()

    while True:
        #for _ in range(5):
        
        s = np.array(env.reset())
        start_time=time.time()
        done = False
        while not done:            
            step_start = time.time()
            
            if user_input == 'e':
                _, sampled_actions, sigma = actor.forward(torch.cuda.FloatTensor(s), play=True)                

                s_prime, r, done, _ = env.step(sampled_actions)
            elif user_input == 's':
                _, sampled_actions, sigma = actor.forward(torch.cuda.FloatTensor(s), play=True)                

                s_prime, r, done, _ = env.step(sampled_actions)

                user_input = user_key_input()
            
            step += 1            
            score += r
            s = s_prime    
            
            time_to_sleep = control_time-(time.time()-step_start)
            if time_to_sleep>0:
                time.sleep(time_to_sleep)            
        
        print('step : {0} | score : {1} | time : {2} s'.format(step, score, time.time()-start_time))            

        step = 0
        score = 0        

        print("1 episode or 1 step? insert 'e' or 's' or 'd'")
        while True:        
            user_input = input()

            if user_input != 'e' and user_input != 's' and user_input != 'd':
                print("Wrong input is inserted. Please insert 'e' or 's' or 'd'")
                continue
            else:
                break

        if user_input == "d":            
            return 0

def user_key_input():
    while True:        
        play_value = input()

        if play_value != 'e' and play_value != 's':
            print("Wrong input is inserted. Please insert 'e' or 's'")
            continue
        else:
            break

    return play_value

if __name__ == '__main__':
    #torch.manual_seed(2017710643)
    torch.manual_seed(3000)
    if args.train:
        train()
    else:
        play()
