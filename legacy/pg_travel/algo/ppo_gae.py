import numpy as np
import torch
from torch.random import initial_seed

gamma = 0.99
lamda = 0.95
batch_size = 128
clip_param = 0.2


def get_gae(rewards, masks, values):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        running_tderror = rewards[t] + gamma * previous_value * masks[t] - \
                        values.data[t]
        running_advants = running_tderror + gamma * lamda * \
                        running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    return returns, advants


def log_density(actions, mu, std):
    actions = torch.Tensor(actions)
    mu = mu.squeeze()
    std = std.squeeze()
    dist = torch.distributions.Normal(mu, std)
    prob = dist.log_prob(actions)

    return prob


def surrogate_loss(actor, advants, states, old_policy, actions, index):
    mu, std = actor(torch.Tensor(states))
    new_policy = log_density(actions, mu, std)
    old_policy = old_policy[index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advants
    return surrogate, ratio


def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])
    values = critic(torch.Tensor(states))

    ## GAE
    returns, advants = get_gae(rewards, masks, values)
    mu, std = actor(torch.Tensor(states))
    old_policy = log_density(actions, mu, std)
    old_values = critic(torch.Tensor(states))

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    ## update
    for epoch in range(10):
        np.random.shuffle(arr)

        for i in range(n // batch_size):
            batch_index = arr[batch_size * i:batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)

            ## Critic Loss
            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -clip_param,
                                         clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            ## Actor Loss
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_param,
                                        1.0 + clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            total_loss = actor_loss + 0.5 * critic_loss

            critic_optim.zero_grad()
            total_loss.backward(retain_graph=True)
            critic_optim.step()

            actor_optim.zero_grad()
            total_loss.backward()
            actor_optim.step()