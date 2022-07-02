from this import s
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch import from_numpy, no_grad, save, load, tensor, clamp
from torch import float as torch_float
from torch import long as torch_long
from torch import min as torch_min
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from torch import manual_seed
from collections import namedtuple
import torch

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class PPOAgent:
    """
    PPOAgent implements the PPO RL algorithm (https://arxiv.org/abs/1707.06347).
    It works with a set of discrete actions.
    It uses the Actor and Critic neural network classes defined below.
    """

    def __init__(self, numberOfInputs=None, numberOfActorOutputs=None, clip_param=0.2, max_grad_norm=0.5, ppo_update_iters=5,
                 batch_size=8, gamma=0.99, use_cuda=False, actor_lr=0.001, critic_lr=0.003, seed=None):
        super().__init__()
        if seed is not None:
            manual_seed(seed)

        # Hyper-parameters
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ppo_update_iters = ppo_update_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.use_cuda = use_cuda

        # models
        self.actor_net = Actor(numberOfInputs, numberOfActorOutputs)
        self.critic_net = Critic(numberOfInputs)

        if self.use_cuda:
            self.actor_net.cuda()
            self.critic_net.cuda()

        # Create the optimizers
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), actor_lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), critic_lr)

        # Training stats
        self.buffer = []

    def work(self, agentInput, type_="selectAction"):
        """
        Forward pass of the PPO agent. Depending on the type_ argument, it either explores by sampling its actor's
        softmax output, or eliminates exploring by selecting the action with the maximum probability (argmax).

        :param agentInput: The actor neural network input vector
        :type agentInput: vector
        :param type_: "selectAction" or "selectActionMax", defaults to "selectAction"
        :type type_: str, optional
        """
        for i in range(2):
            # agentInput[i] = from_numpy(np.array(agentInput[i])).float().unsqueeze(0)  # Add batch dimension with unsqueeze
            agentInput[i] = from_numpy(np.array(agentInput[i])).float().unsqueeze(0)  # Add batch dimension with unsqueeze
        
        if self.use_cuda:
            for i in range(2):
                agentInput[i] = agentInput[i].cuda()

        with no_grad():
            mu, sigma = self.actor_net(agentInput[0], agentInput[1])
            # print(mu, sigma)
            mu, sigma = mu.squeeze(dim=0), sigma.squeeze(dim=0)
            dis = torch.distributions.MultivariateNormal(mu, torch.diag_embed(sigma))   # 构建分布
            action = dis.sample()   # 采样出一个动作
            # print(type(action), action.shape)
            action_log_prob = dis.log_prob(action)  # 计算动作的概率
        
        return action, action_log_prob

        # if type_ == "selectAction":
        #     c = Categorical(action_prob)
        #     action = c.sample()
        #     return action.item(), action_prob[:, action.item()].item()
        # elif type_ == "selectActionMax":
        #     return np.argmax(action_prob).item(), 1.0

    def save(self, path):
        """
        Save actor and critic models in the path provided.

        :param path: path to save the models
        :type path: str
        """
        save(self.actor_net.state_dict(), path + '_actor.pkl')
        save(self.critic_net.state_dict(), path + '_critic.pkl')

    def load(self, path):
        """
        Load actor and critic models from the path provided.

        :param path: path where the models are saved
        :type path: str
        """
        actor_state_dict = load(path + '_actor.pkl')
        critic_state_dict = load(path + '_critic.pkl')
        self.actor_net.load_state_dict(actor_state_dict)
        self.critic_net.load_state_dict(critic_state_dict)

    def storeTransition(self, transition):
        """
        Stores a transition in the buffer to be used later.

        :param transition: contains state, action, action_prob, reward, next_state
        :type transition: namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
        """
        self.buffer.append(transition)

    def trainStep(self, batchSize=None):
        """
        Performs a training step for the actor and critic models, based on transitions gathered in the
        buffer. It then resets the buffer.

        :param batchSize: Overrides agent set batch size, defaults to None
        :type batchSize: int, optional
        """
        # Default behaviour waits for buffer to collect at least one batch_size of transitions
        if batchSize is None:
            if len(self.buffer) < self.batch_size:
                return
            batchSize = self.batch_size

        # Extract states, actions, rewards and action probabilities from transitions in buffer
        # state_img = torch.cat([t.state[0].unsqueeze(0) for t in self.buffer], 0)
        # state_imu = torch.cat([t.state[1].unsqueeze(0) for t in self.buffer], 0)
        state_img = torch.cat([t.state[0] for t in self.buffer], 0)
        state_imu = torch.cat([t.state[1] for t in self.buffer], 0)

        action = torch.cat([t.action.unsqueeze(0) for t in self.buffer], 0)  #TODO: check if this is correct
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = tensor([t.a_log_prob for t in self.buffer], dtype=torch_float).view(-1, 1)

        # Unroll rewards
        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = tensor(Gt, dtype=torch_float)

        # Send everything to cuda if used
        if self.use_cuda:
            state_img, state_imu, action, old_action_log_prob = state_img.cuda(), state_imu.cuda(), action.cuda(), old_action_log_prob.cuda()
            Gt = Gt.cuda()

        # Repeat the update procedure for ppo_update_iters
        for i in range(self.ppo_update_iters):
            # Create randomly ordered batches of size batchSize from buffer
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), batchSize, False):
                # Calculate the advantage at each step
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state_img[index], state_imu[index])
                delta = Gt_index - V
                advantage = delta.detach()

                # Get the current probabilities
                # Apply past actions with .gather()
                # TODO: prob
                # action_prob = self.actor_net(state_img[index], state_imu[index]).gather(1, action[index])  # new policy
                mu, sigma = self.actor_net(state_img[index], state_imu[index])
                # action_prob = []
                # for i in range(mu.shape[0]):
                #     dis = torch.distributions.MultivariateNormal(mu[i], torch.diag_embed(sigma[i]))
                #     action_prob.append(dis.log_prob(action[i]))

                # print(torch.diag_embed(sigma))    # check: ok
                # seems like this can generate a list of distributions
                dis = torch.distributions.MultivariateNormal(mu, torch.diag_embed(sigma))
                action_prob = dis.log_prob(action[index]).view(-1, 1)

                # print("action prob shape", action_prob.shape)

                # PPO
                ratio = (action_prob / old_action_log_prob[index])  # Ratio between current and old policy probabilities
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch_min(surr1, surr2).mean()  # MAX->MIN descent
                self.actor_optimizer.zero_grad()  # Delete old gradients
                action_loss.backward()  # Perform backward step to compute new gradients
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)  # Clip gradients
                self.actor_optimizer.step()  # Perform training step based on gradients

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        # After each training step, the buffer is cleared
        del self.buffer[:]


class Actor(nn.Module):
    """
    Actor network for the actor-critic algorithm.
    """
    def __init__(self, numberOfInputs=None, numberOfOutputs=None):
        super(Actor, self).__init__()
        # img
        self.img_channel = nn.Sequential(
            # 3 x 240 x 400
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            # 8 x 60 x 100
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            # 8 x 15 x 25
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 5), stride=4),
            # 1 x 4 x 6
        )
        # imu
        self.imu_channel = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 12),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(24 + 12, 4)
        self.fc2 = nn.Linear(24 + 12, 4)

    def forward(self, img, imu):
        # img
        img = self.img_channel(img)
        img = img.view(img.shape[0], -1) # 24
        # 1 x 24
        # imu
        imu = self.imu_channel(imu) # 12
        imu = imu.view(imu.shape[0], -1)
        # 1 x 12
        # print(img.shape, imu.shape)
        # cat
        img_imu = torch.cat((img, imu), 1)
        # sigma & mu
        mu = torch.tanh(self.fc1(img_imu)) * 100 + 100   # mu[4], [0, 200]
        sigma = F.softplus(self.fc2(img_imu)) + 0.001      # sigma[4], >0
        # print(sigma.shape)
        # print(mu.shape)

        return mu, sigma


class Critic(nn.Module):
    """
    Critic network for the actor-critic algorithm.
    """
    def __init__(self, numberOfInputs=None):
        super(Critic, self).__init__()
        # img
        self.img_channel = nn.Sequential(
            # 3 x 240 x 400
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            # 8 x 60 x 100
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            # 8 x 15 x 25
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 5), stride=4),
            # 1 x 4 x 6
        )
        # imu
        self.imu_channel = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 12),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(24 + 12, 1)

    def forward(self, img, imu):
        # img
        img = self.img_channel(img)
        img = img.view(img.shape[0], -1) # 24
        # 1 x 24
        # imu
        imu = self.imu_channel(imu) # 12
        imu = imu.view(imu.shape[0], -1)
        # 1 x 12
        # print(img.shape, imu.shape)
        # cat
        img_imu = torch.cat((img, imu), 1)
        # value
        value = self.fc1(img_imu)

        return value
