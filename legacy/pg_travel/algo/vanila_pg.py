import numpy as np
import torch

gamma = 0.99
batch_size = 64


def get_returns(rewards, masks):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)

    running_returns = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        returns[t] = running_returns

    returns = (returns - returns.mean()) / returns.std()
    return returns


def get_loss(actor, returns, states, actions):
    mu, std = actor(torch.Tensor(states))
    dist = torch.distributions.Normal(mu, std)
    log_probs = dist.log_prob(torch.Tensor(actions))

    objective = returns * log_probs
    objective = objective.mean()

    return -objective


def train_model(actor, critic, memory, actor_optim, ciritic_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = np.vstack(memory[:, 1])
    rewards = np.vstack(memory[:, 2])
    masks = np.vstack(memory[:, 3])

    returns = get_returns(rewards, masks)
    train_ciritic(critic, states, returns, ciritic_optim)
    train_actor(actor, returns, states, actions, actor_optim)


def train_ciritic(critic, states, returns, critic_optim):
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for epoch in range(5):
        np.random.shuffle(arr)

        for i in range(n // batch_size):
            batch_index = arr[batch_size * i:batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            target = returns[batch_index]

            values = critic(inputs)
            loss = criterion(values, target)
            critic_optim.zero_grad()
            loss.backward()
            critic_optim.step()


def train_actor(actor, returns, states, actions, actor_optim):
    loss = get_loss(actor, returns, states, actions)
    actor_optim.zero_grad()
    loss.backward()
    actor_optim.step()