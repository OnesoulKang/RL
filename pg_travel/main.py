import torch
import numpy as np
import torch.optim as optim
import gym
from model import Actor, Critic
from collections import deque
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default='ppo')
args = parser.parse_args()

if args.algo == 'pg':
    from algo.vanila_pg import train_model
elif args.algo == 'ppo':
    from algo.ppo_gae import train_model

import matplotlib.pyplot as plt

lr = 0.0003


def get_actions(mu, std):
    dist = torch.distributions.Normal(mu, std)
    actions = dist.sample()

    return actions.data.numpy()


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    print('state size :', num_inputs, '\naction size :', num_actions)

    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)

    actor_optim = optim.Adam(actor.parameters(), lr=lr)
    critic_optim = optim.Adam(critic.parameters(), lr=lr)

    scores = []
    prev_score = 0

    for episode in range(3000):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        score = 0
        done = False
        state = env.reset()

        # while steps < 1000:
        #     episodes += 1
        #     state = env.reset()
        #     score = 0

        #     for _ in range(1000):
        while not done:
            steps += 1
            mu, std = actor(torch.Tensor(state))
            action = get_actions(mu, std)[0]
            next_state, reward, done, _ = env.step([action])

            if done:
                mask = 0
            else:
                mask = 1

            memory.append([state, action, reward, mask])

            score += reward
            state = next_state

            if done:
                break

        scores.append(score * 0.6 + prev_score * 0.4)
        prev_score = score
        # score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episode, score))

        actor.train(), critic.train()
        train_model(actor, critic, memory, actor_optim, critic_optim)

    plt.plot(scores)
    plt.savefig("matplotlib.png")
