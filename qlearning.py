import random

import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers


class QNetwork(models.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu', input_shape=(state_size,))
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(action_size, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        q_values = self.output_layer(x)
        return q_values


def q_learning_train(model, target_model, gamma, batch_size, optimizer, replay_memory):
    if len(replay_memory) < batch_size:
        return

    minibatch = np.array(random.sample(replay_memory, batch_size))
    states = np.vstack(minibatch[:, 0])
    actions = minibatch[:, 1].astype(int)
    rewards = minibatch[:, 2]
    next_states = np.vstack(minibatch[:, 3])
    dones = minibatch[:, 4]

    target = rewards + gamma * np.max(target_model.predict(next_states), axis=1) * (1 - dones)

    with tf.GradientTape() as tape:
        q_values = model(states)
        selected_action_values = tf.reduce_sum(tf.one_hot(actions, model.output_shape[1]) * q_values, axis=1)
        loss = tf.reduce_mean(tf.square(selected_action_values - target))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
