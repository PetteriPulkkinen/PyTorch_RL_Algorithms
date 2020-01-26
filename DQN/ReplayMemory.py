import torch
import numpy as np


class ReplayMemory(object):
    def __init__(self, n_obs, max_size):
        self.max_size = max_size
        self.size = 0
        self._n_obs = n_obs

        self.act = torch.empty((self.max_size,), dtype=torch.long)
        self.state = torch.empty((self.max_size, n_obs))
        self.reward = torch.empty((self.max_size,))
        self.next_state = torch.empty((self.max_size, n_obs))
        self.done = torch.empty((self.max_size,))

    def sample(self, n_batch):
        idx = np.random.choice(np.arange(self.size), size=(n_batch,))

        return (self.act[idx].view(n_batch, 1),
                self.state[idx].view(n_batch, self._n_obs),
                self.reward[idx].view(n_batch, 1),
                self.next_state[idx].view(n_batch, self._n_obs),
                self.done[idx].view(n_batch, 1))

    def store(self, act, state, reward, next_state, done):
        idx = self.size % self.max_size
        self.act[idx] = int(act)
        self.state[idx, :] = torch.Tensor(state)
        self.reward[idx] = reward
        self.next_state[idx, :] = torch.Tensor(next_state)
        self.done[idx] = float(done)

        if self.size < self.max_size:
            self.size += 1


if __name__ == '__main__':
    rm = ReplayMemory(4, 10000)

    for i in range(1000):

        act = np.random.randint(10)
        state = np.random.randn(4)
        next_state = np.random.randn(4)
        reward = np.random.randn()
        done = bool(np.random.randint(2))
        rm.store(act, state, reward, next_state, done)

    import time
    start = time.time()

    for i in range(1000):
        exp = rm.sample(32)

    print(1000/(time.time() - start))
