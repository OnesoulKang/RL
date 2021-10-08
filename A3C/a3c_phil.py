## https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py

import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from shared_adam import SharedAdam


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(input_dims, 128)
        self.v1 = nn.Linear(input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.rewards = []
        self.actions = []
        self.states = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)

        R = v[-1] * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns - values)**2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_prob = dist.log_prob(actions)
        actor_loss = -log_prob * (returns - values).detach()

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        pi, _ = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action


class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                 gamma, lr, name, global_ep_idx, r_que, env_id):
        super(Agent, self).__init__()

        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.r_que = r_que
        self.env = gym.make(env_id)
        self.optimizer = optimizer

    def run(self):
        t_step = 1

        while self.episode_idx.value < N_GAMES:
            done = False
            observation = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()

            while not done:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, _ = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % T_MAX == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()

                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()

                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()

                t_step += 1
                observation = observation_

            with self.episode_idx.get_lock():
                self.episode_idx.value += 1

            self.r_que.put(score)

            print(self.name, 'episode ', self.episode_idx.value,
                  'reward %.1f' % score)

        self.r_que.put(None)


if __name__ == '__main__':
    lr = 1e-4
    env_id = 'CartPole-v0'
    input_dims = 4
    n_actions = 2
    N_GAMES = 3000
    T_MAX = 5

    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    global_ep, r_que = mp.Value('i', 0), mp.Queue()
    test_R = []

    optim = SharedAdam(global_actor_critic.parameters(),
                       lr=lr,
                       betas=(0.92, 0.999))

    workers = [
        Agent(global_actor_critic,
              optim,
              input_dims,
              n_actions,
              gamma=0.99,
              lr=lr,
              name=i,
              global_ep_idx=global_ep,
              r_que=r_que,
              env_id=env_id) for i in range(mp.cpu_count())
    ]

    [w.start() for w in workers]

    res = []
    while True:
        r = r_que.get()
        if r is not None:
            res.append(r)
        else:
            break

    [w.join() for w in workers]

    plt.plot(res)
    plt.savefig("matplotlib.png")
    # plt.show()