import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN:
    def __init__(self, state_space_shape, num_actions, model, target_model, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = model
        self.target_model = target_model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def load(self, model_name, episode):
        self.model.load_state_dict(torch.load(f'dqn_{model_name}_{episode}.pt'))

    def save(self, model_name, episode):
        torch.save(self.model.state_dict(), f'dqn_{model_name}_{episode}.pt')

    def train(self):
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            target = self.model(state_tensor).detach().clone().squeeze()
            if done:
                target[action] = reward
            else:
                with torch.no_grad():
                    max_future_q = torch.max(self.target_model(next_state_tensor)).item()
                target[action] = reward + self.discount_factor * max_future_q

            states.append(state)
            targets.append(target)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        targets_tensor = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()


class DDQN(DQN):
    def train(self):
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            target = self.model(state_tensor).detach().clone().squeeze()
            if done:
                target[action] = reward
            else:
                with torch.no_grad():
                    next_q_values = self.model(next_state_tensor)
                    best_action = torch.argmax(next_q_values)
                    max_q = self.target_model(next_state_tensor)[0][best_action].item()
                target[action] = reward + self.discount_factor * max_q

            states.append(state)
            targets.append(target)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        targets_tensor = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()


class DuelingDQNModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQNModel, self).__init__()
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate the size after conv layers to feed into linear layer
        dummy_input = torch.zeros(1, *input_shape)
        conv_out_size = self.features(dummy_input).view(1, -1).size(1)

        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))



class DuelingDQN:
    def __init__(self, state_space_shape, num_actions, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = DuelingDQNModel(state_space_shape, num_actions)
        self.target_model = DuelingDQNModel(state_space_shape, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def load(self, model_name, episode):
        self.model.load_state_dict(torch.load(f'duelingdqn_{model_name}_{episode}.pt'))

    def save(self, model_name, episode):
        torch.save(self.model.state_dict(), f'duelingdqn_{model_name}_{episode}.pt')

    def train(self):
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            target = self.model(state_tensor).detach().clone().squeeze()
            if done:
                target[action] = reward
            else:
                with torch.no_grad():
                    max_future_q = torch.max(self.target_model(next_state_tensor)).item()
                target[action] = reward + self.discount_factor * max_future_q

            states.append(state)
            targets.append(target)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        targets_tensor = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()