import gym
import random


def random_action():
    return random.choice([0, 1, 2, 3])


class Agent:
    def __init__(self):
        self.alpha = 1
        self.gama = 1
        self.epsilon = 0.5
        self.Q_table = {}

    def add(self, state_):
        if state_ not in self.Q_table:
            self.Q_table[state_] = {}
            for i in range(4):
                self.Q_table[state_][i] = 0

    def update(self, ps, pa, s, r, d):
        if d:
            self.Q_table[ps][pa] = (1 - self.alpha) * self.Q_table[ps][pa] + self.gama * r
            return
        max_q = -10000
        for i in range(4):
            if self.Q_table[s][i] > max_q:
                max_q = self.Q_table[s][i]
        self.Q_table[ps][pa] = (1 - self.alpha) * self.Q_table[ps][pa] + self.gama * (r + max_q)

    def epsilon_greedy(self, s):
        max_action = []
        other_action = []
        max_q = -10000
        for i in range(4):
            if self.Q_table[s][i] > max_q:
                max_q = self.Q_table[s][i]
        for i in range(4):
            if self.Q_table[s][i] == max_q:
                max_action.append(i)
            else:
                other_action.append(i)
        try:
            if random.random() > self.epsilon:
                act = random.choice(max_action)
            else:
                act = random.choice(other_action)
        except:
            act = random.choice(max_action)
        return act

    def best_action(self, s):
        max_action = []
        max_q = -10000
        for i in range(4):
            if self.Q_table[s][i] > max_q:
                max_q = self.Q_table[s][i]
        for i in range(4):
            if self.Q_table[s][i] == max_q:
                max_action.append(i)
        return random.choice(max_action)


# initialization
env = gym.make("CliffWalking-v0")
state = env.reset()
agent = Agent()
agent.add(state)

# train
for t in range(5000):
    while True:
        if t == 1000:
            agent.epsilon = 0.1
        elif t == 3000:
            agent.epsilon = 0.01
        action = agent.epsilon_greedy(state)
        pre_state = state
        state, reward, done, info = env.step(action)
        if not done:
            agent.add(state)
        agent.update(pre_state, action, state, reward, done)
        if done:
            state = env.reset()
            break

# test
val = 0
cnt = 0
for _ in range(1000):
    while True:
        action = agent.best_action(state)
        state, reward, done, info = env.step(action)
        val = val + reward
        if done:
            state = env.reset()
            cnt = cnt + 1
            break
print(val / cnt)

# check Q_table
for i in range(len(agent.Q_table)):
    print(i, agent.Q_table[i])
