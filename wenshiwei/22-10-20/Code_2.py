import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import random

alpha = 0.01
epsilon = 0.1
gamma = 0.9
batch_size = 32
n_record = 2000
n_change = 200
env = gym.make("CartPole-v0")
n_action = env.action_space.n
n_state = env.observation_space.shape[0]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(n_state, 50)
        self.fc.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, n_action)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s):
        s = self.fc(s)
        s = f.relu(s)
        return self.out(s)


class Agent:
    def __init__(self):
        self.q_net = Net()
        self.t_net = Net()
        self.loss_f = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=alpha)
        self.record = np.zeros((n_record, n_state * 2 + 2))
        self.record_cnt = 0
        self.learn_cnt = 0

    def experience(self, s, a, r, s_):
        self.record[self.record_cnt % n_record, :] = np.hstack((s, a, r, s_))
        self.record_cnt += 1

    def epsilon_greedy(self, s):
        s = torch.FloatTensor(s)
        if random.random() > epsilon:
            q_s_a = self.q_net.forward(s)
            act = torch.max(q_s_a, 0)[1].data.numpy()
        else:
            act = random.randint(0, n_action - 1)
        return act

    def learn(self):
        if self.learn_cnt % n_change == 0:
            self.t_net.load_state_dict(self.q_net.state_dict())
        self.learn_cnt += 1
        samples = np.array(random.choices(self.record, k=batch_size))
        ps = torch.FloatTensor(samples[:, :n_state])
        pa = torch.LongTensor(samples[:, n_state: n_state + 1])
        pr = torch.FloatTensor(samples[:, n_state + 1: n_state + 2])
        ps_ = torch.FloatTensor(samples[:, -n_state:])

        q_s = self.q_net.forward(ps).gather(1, pa)
        q_s_ = self.t_net.forward(ps_).detach()
        q_target = pr + gamma * q_s_.max(1)[0].view(batch_size, 1)
        loss = self.loss_f(q_s, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


agent = Agent()

for i in range(400):
    state = env.reset()
    ep_r = 0
    while True:
        env.render()
        action = agent.epsilon_greedy(state)
        state_, reward, done, info = env.step(action)
        x, _, theta, _ = state_
        reward = (env.x_threshold - abs(x)) / env.x_threshold / 2 \
                 + (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians / 2
        agent.experience(state, action, reward, state_)
        ep_r += reward
        if agent.record_cnt > n_record:
            agent.learn()
            if done:
                print("EP:  ", i, "  |  ", ep_r)
        if done:
            break
        state = state_

