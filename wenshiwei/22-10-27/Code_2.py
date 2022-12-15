import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('CartPole-v0')
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
n_episode = 5000
gamma = 0.95
alpha = 1e-3


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.action_layer = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        self.value_layer = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        act = F.softmax(self.action_layer(s), dim=-1)
        val = self.value_layer(s)
        return act, val


class AC(nn.Module):
    def __init__(self):
        super(AC, self).__init__()
        self.model = Net()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        probs, _ = self.model(s)
        act = torch.multinomial(probs, 1)
        return act.item(), probs

    def critic_learn(self, s, s_, r, done):
        s = torch.FloatTensor(s)
        s_ = torch.FloatTensor(s_)
        r = torch.FloatTensor([r])

        _, v = self.model(s)
        _, v_ = self.model(s_)
        v_ = v_.detach()

        target = r
        if not done:
            target += gamma * v_

        loss_func = nn.MSELoss()
        loss = loss_func(v, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        advantage = (target - v).detach()
        return advantage

    def actor_learn(self, advantage, s, a):
        _, probs = self.choose_action(s)
        log_prob = probs.log()[a]

        loss = -advantage * log_prob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


agent = AC()

x, y = [], []
ep_reward = 0
for episode in range(n_episode):
    state = env.reset()
    while True:
        action, _ = agent.choose_action(state)
        state_, reward, done, _ = env.step(action)
        ep_reward += reward

        advantage = agent.critic_learn(state, state_, reward, done)
        agent.actor_learn(advantage, state, action)

        state = state_
        if done:
            break
    if episode % 100 == 0:
        print("EP:  ", episode, "  |  ", ep_reward / 100)
        ep_reward = 0

    x.append(episode)
    y.append(ep_reward)

