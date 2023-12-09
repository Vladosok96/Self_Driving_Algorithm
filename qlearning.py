import numpy as np
# from tensorflow.keras import layers, models, optimizers
import functions


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from collections import namedtuple, deque
import math


"""
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  # Опыт агента (состояние, действие, награда, новое состояние, завершено)
        self.gamma = 0.95  # Фактор дисконтирования
        self.epsilon = 1.0  # Исследование против эксплуатации
        self.epsilon_decay = 0.995  # Уменьшение исследования с течением времени
        self.epsilon_min = 0.01  # Минимальное значение исследования
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(optimizer=optimizers.Adam(lr=self.learning_rate), loss='mse', metrics='mae')

        print(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state):
        if len(self.memory) > 100000:
            print("reduce memory...")
            self.memory = functions.custom_random_choice(self.memory, 10000, replace=False)
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, self.action_size)
        q_values = self.model.predict(np.array([state]))
        return q_values[0]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = functions.custom_random_choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target  # Обновление Q-значения для соответствующего действия
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0, batch_size=batch_size)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
"""


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Пример: один скрытый слой с 64 нейронами
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class QLearningAgent:
    def __init__(self, input_size, output_size, learning_rate=0.001, gamma=0.99):
        self.q_network = QNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def update_q_values(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Прямой проход
        current_q_values = self.q_network(state)
        try:
            current_q_value = current_q_values[action]
        except:
            return 0

        # Вычисление целевого Q-значения с использованием формулы Q-learning
        with torch.no_grad():
            next_q_values = self.q_network(next_state)
            # max_next_q_value, _ = torch.max(next_q_values, dim=0, keepdim=True)
            target_q_value = reward + self.gamma * next_q_values

        # Рассчет loss и обновление весов
        loss = self.criterion(current_q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = 2
# Get the number of state observations
state, info = np.array
n_observations = len(state)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
