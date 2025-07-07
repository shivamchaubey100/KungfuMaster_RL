import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
import helper  
from memory import ReplayBuffer

class DQNAgent:
    def __init__(
        self,
        input_shape,
        num_actions,
        gamma=0.99,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=1_000_000,
        target_update_freq=10_000,
    ):
        # Environment / action parameters
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.possible_actions = list(range(num_actions))

        # Discount factor
        self.gamma = gamma

        # Epsilon‑greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        # Learning rate for both helper compilation and optimizer
        self.learn_rate = lr

        # Build main and target networks using your helper
        self.model = helper.KungFu(self)
        self.target_model = helper.KungFu(self)
        # Sync initial weights
        self.target_model.set_weights(self.model.get_weights())

        # Optimizer & loss for gradient updates
        self.optimizer = Adam(self.learn_rate)
        self.loss_fn = tf.keras.losses.Huber()

        # Target network update cadence
        self.target_update_freq = target_update_freq
        self.step_count = 0

    def act(self, state):
        """Epsilon‑greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.possible_actions)
        q_values = self.model(np.expand_dims(state, 0), training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def train_step(self, batch):
        """Perform one optimization step on a sampled batch."""
        states, actions, rewards, next_states, dones = batch

        # Compute target Q-values
        next_q = self.target_model(next_states, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1)
        target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            q_preds = self.model(states, training=True)
            # pick the Q-values for the actions taken
            indices = tf.stack([tf.range(actions.shape[0]), actions], axis=1)
            chosen_q = tf.gather_nd(q_preds, indices)
            loss = self.loss_fn(target_q, chosen_q)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss.numpy()

    def update_epsilon(self):
        """Linearly decay epsilon until epsilon_end."""
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay

    def update_target_network(self):
        """Copy weights from main network to the target network."""
        self.target_model.set_weights(self.model.get_weights())

    def maybe_update_target(self):
        """Update target network every target_update_freq steps."""
        if self.step_count and self.step_count % self.target_update_freq == 0:
            self.update_target_network()
