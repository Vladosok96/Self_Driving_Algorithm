import numpy as np
from keras import layers, models
import functions


# Машинное обучение
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
            layers.Dense(4, input_shape=(self.state_size,), activation='relu'),
            layers.Dense(3, activation='relu'),
            layers.Dense(2, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse', metrics='mae')

        print(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state):
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
            target_f[0] = action * target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0, batch_size=16)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay