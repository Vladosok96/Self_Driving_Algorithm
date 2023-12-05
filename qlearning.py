import numpy as np


# Класс для Q-обучения, специфичный для вашей задачи
class QLearningAgent:
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Создаем Q-таблицу и инициализируем ее случайными значениями
        self.q_table = np.random.rand(n_actions)

    def choose_action(self, state):
        # Принимаем случайное решение с вероятностью exploration_rate
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.n_actions)
        # В противном случае выбираем действие с максимальным Q-значением для данного состояния
        return np.argmax(self.q_table)

    def update_q_table(self, state, action, reward, next_state):
        # Обновляем Q-значение в соответствии с формулой Q-обучения
        old_value = self.q_table[action]
        next_max = np.max(self.q_table)
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[action] = new_value
