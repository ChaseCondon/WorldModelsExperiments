from collections import deque
from random import sample

class ReplayBuffer(object):
    """Implementation of the DQN's buffer for experience replay"""

    def __init__(self, buffer_max=10000):
        self.buffer = deque()
        self.buffer_max = buffer_max
        self.size = 0

    def add(self, s, a, r, d, s2):
        experience = (s, a, r, d, s2)
        self.buffer.append(experience)

        if len(self.buffer) > self.buffer_max:
            self.buffer.popleft()
        else:
            self.size += 1

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch = sample(self.buffer, self.size)
        else:
            batch = sample(self.buffer, batch_size)

        return batch
