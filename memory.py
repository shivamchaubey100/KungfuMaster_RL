# memory.py

import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.memory = deque(maxlen=max_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Sample a batch of transitions."""
        batch = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
