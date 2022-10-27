import gym
import torch
import torch.nn as nn
import torch.nn.functional as f

alpha = 1e-3
gamma = 0.98
env = gym.make("CartPole-v0")
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden = 128


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.fc = nn.Linear(n_state, n_hidden)
        self.out = nn.Linear(n_hidden, n_action)

    def forward(self, s):
        s = f.relu(self.fc(s))
        return f.softmax(self.out(s), dim=0)


class Agent:
    def __init__(self):
        self.policy = PNet()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)

    def take_action(self, s):
        s = torch.FloatTensor(s)
        sp = self.policy(s)
        action_dist = torch.distributions.Categorical(sp)
        return action_dist.sample().item()

    def update(self, ep_d):
        r_list = ep_d["rewards"]
        a_list = ep_d["actions"]
        s_list = ep_d["states"]
        g = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(r_list))):
            r = r_list[i]
            s = torch.FloatTensor(s_list[i])
            a = torch.tensor(a_list[i])
            log_prob = torch.log(self.policy(s).gather(-1, a))
            g = gamma * g + r
            loss = -log_prob * g
            loss.backward()
        self.optimizer.step()


agent = Agent()
ep_r = 0
# env.render()
for t in range(5000):
    ep_data = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
    }
    done = False
    state = env.reset()
    while not done:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        ep_data["states"].append(state)
        ep_data["next_states"].append(next_state)
        ep_data["rewards"].append(reward)
        ep_data["actions"].append(action)
        state = next_state
        ep_r += reward
    agent.update(ep_data)
    if t % 100 == 0:
        print("EP:  ", t, "  |  ", ep_r / 100)
        ep_r = 0
