import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
import functions

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
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, batch_size):
        self.memory.append([state, action, reward, next_state, done])
        # self.memory = self.memory[-batch_size:]

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.array([state]))
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = functions.custom_random_choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0]))
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Пример использования:
state_size = 2  # Размерность состояния (координаты, скорость, угол поворота руля)
action_size = 2  # Количество действий (ускорение, угол поворота руля)
agent = DQNAgent(state_size, action_size)

# Предполагается, что у вас есть функция get_state(), которая возвращает текущее состояние автомобиля.
# Предполагается, что у вас есть функции perform_action() и get_reward(), которые выполняют действие и возвращают награду соответственно.

num_episodes = 1000
for episode in range(num_episodes):
    # state = get_state()
    state = [0, 1]
    total_reward = 0
    done = False
    counter = 0
    while not done:
        action = agent.act(state)
        next_state = state.copy()
        reward = 0

        # perform action
        if state[1] == 1:

            if action == 1:
                reward = 1
            else:
                reward = -1

            next_state[0] += 0.1
            if state[0] >= 0.8:
                next_state[1] = 0
        else:

            if action == 0:
                reward = 1
            else:
                reward = -1

            next_state[0] -= 0.1
            if state[0] <= 0.2:
                next_state[1] = 1

        print("reward:", reward, "total_reward:", total_reward, "action", action, "state:", state)
        agent.remember(state.copy(), action, reward, next_state.copy(), done, batch_size=32)
        state = next_state
        total_reward += reward
        if done:
            print("Эпизод: {}, Награда: {}, Исследование: {:.2}".format(episode + 1, total_reward, agent.epsilon))
        counter += 1
        if counter > 32:
            agent.replay(batch_size=32)
            counter = 0
