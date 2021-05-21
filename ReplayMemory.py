import random
import const
from collections import namedtuple, deque


Experience = namedtuple('Experience', 'prev_state action current_state reward')


class ReplayMemory:
    def __init__(self, capacity: int = const.REPLAY_CAPACITY):
        self._memory = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self._memory.append(experience)

    def sample(self, batch_size: int = const.REPLAY_BATCH_SIZE):
        if len(self._memory) < batch_size:
            return False, []
        return True, random.sample(self._memory, batch_size)

    def __len__(self):
        return len(self._memory)
