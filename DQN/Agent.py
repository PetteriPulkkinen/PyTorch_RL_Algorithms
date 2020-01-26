import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

from ReplayMemory import ReplayMemory
from CriticNetwork import CriticNetwork


class DQN(object):

    def __init__(self, env, gamma=0.98, start_epsilon=1, end_epsilon=0.01, decay=500, lr=1e-4, n_batch=32,
                 n_memory=50000, n_update_target=500, start_learning=1000, log_dir=None):
        self.env = env
        n_act = self.env.action_space.n
        n_obs = np.prod(self.env.observation_space.shape)

        self.critic = CriticNetwork(n_act=n_act, n_obs=n_obs)
        self.target_critic = CriticNetwork(n_act=n_act, n_obs=n_obs)
        self._update_target()

        self.memory = ReplayMemory(n_obs, n_memory)

        self.gamma = gamma
        self.s_epsilon = start_epsilon
        self.e_epsilon = end_epsilon
        self.decay = decay
        self.n_batch = n_batch
        self.n_update_target = n_update_target
        self.start_learning = start_learning

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def choose(self, obs, epsilon):
        # Explore with probability epsilon (otherwise continue)
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            t_obs = torch.Tensor(obs).view(1, -1)
            return torch.argmax(self.critic(t_obs).view(-1)).item()

    def learn(self, n_steps=0, log_dir=None):
        writer = SummaryWriter(log_dir)

        i_episode = 0
        i_step = 0

        ep_loss = 0
        ep_reward = 0
        obs = self.env.reset()
        while(i_step < n_steps):

            act = self.choose(obs, self._epsilon(i_step))
            next_obs, reward, done, _ = self.env.step(act)
            self.memory.store(act, obs, reward, next_obs, done)
            ep_reward += reward
            obs = next_obs

            if self.memory.size >= self.n_batch and (self.memory.size >= self.start_learning):
                loss = self._train()
                if i_step % self.n_update_target == 0:
                    self._update_target()
                ep_loss += loss
                writer.add_scalar('losses/loss', loss, i_step)

            if done:
                obs = self.env.reset()
                writer.add_scalar('rewards/ep_reward', ep_reward, i_episode)
                print('-' * 10)
                print('Step {} \n Epsiode {} \n EpReward {} \n EpLoss {} Epsilon {}'.format(
                    i_step, i_episode, ep_reward, ep_loss, self._epsilon(i_step)))
                print('-' * 10)
                i_episode += 1
                ep_reward = 0
                ep_loss = 0

            i_step += 1
            writer.add_scalar('rewards/reward', reward, i_step)

        # End training
        writer.close()

    def _epsilon(self, i_step):
        return self.e_epsilon + (self.s_epsilon - self.e_epsilon) * np.exp(-i_step/self.decay)

    def _train(self):
        act, state, reward, next_state, done = self.memory.sample(self.n_batch)

        self.target_critic.eval()
        self.critic.eval()

        # Find the target q values
        next_q_values = self.target_critic(next_state)
        next_q = next_q_values.max(1)[0].view(-1, 1)
        target_q = reward + self.gamma * next_q * (1 - done)

        self.critic.train()

        # Evaluate with the prediction network
        eval_q = self.critic(state).gather(1, act)

        # Calculate loss
        loss = self.criterion(eval_q, target_q)

        self.critic.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def save(self):
        os.makedirs('dqn_agent', exist_ok=True)
        torch.save(self.critic.state_dict(), 'dqn_agent.pth')

    def load(self):
        sd = torch.load('dqn_agent.pth')
        self.critic.load_state_dict(sd)
        self.target_critic.load_state_dict(sd)


if __name__ == '__main__':
    import gym
    import radar_gym
    import matplotlib.pyplot as plt

    env = gym.make('Tracking-v1')
    dqn = DQN(
        env,
        gamma=0.01,
        n_memory=50000,
        n_update_target=100,
        start_epsilon=1,
        end_epsilon=0.01,
        decay=1000,
        lr=1e-4,
        start_learning=1000
    )

    dqn.load()
    dqn.learn(n_steps=10000)
    dqn.save()

    obs = env.reset()
    for i in range(100):
        act = dqn.choose(obs, epsilon=0)
        obs, reward, done, info = env.step(act)
        env.render()
        if done:
            env.reset()
        plt.pause(0.2)
