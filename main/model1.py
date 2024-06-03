import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply linear layer and activation function
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = '../models'
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        if isinstance(state, list):
            state = np.array(state)
        if isinstance(next_state, list):
            next_state = np.array(next_state)
        if isinstance(action, list):
            action = np.array(action)
        if isinstance(reward, list):
            reward = np.array(reward)
        
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # only 1 dimension,  need (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: Get Predicted Q values w curr state
        pred = self.model(state)
        # 2: Q_new = r + y* max(next_Q) -> only if not done
            # pred.clone()
            # preds[argmax(action)] = Q_new
        target = pred.clone()
        for index in range(len(done)):
            Q_new = reward[index]
            if not done[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            target[index][torch.argmax(action).item()] = Q_new

        # 3: optimize
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
