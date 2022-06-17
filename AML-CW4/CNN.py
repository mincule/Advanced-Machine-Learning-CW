from utils import prepare_data, make_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import time
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

torch.manual_seed(1)

# Model: Convolutional Neural Network with kernel size=5, channels=[3,3]
class CNN(nn.Module):
    def __init__(self, channels=[3,3]):
        super().__init__()
        self.chan = channels
        self.dropout = nn.Dropout(0.5)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.chan[0], kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(self.chan[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.chan[0], self.chan[1], kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(self.chan[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
           )
        self.fc = nn.Sequential(
            nn.Linear(self.chan[1]*4389, 120),
            nn.ReLU(),
            self.dropout,
            nn.Linear(120, 84),
            nn.ReLU(),
            self.dropout,
            nn.Linear(84, 63),
           )
        
    def forward(self, x):
        # input shape: [batch_size, 1, 240, 320]
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Train function
def train(batch_size, channels, lr, n_epochs):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = CNN(channels).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loop preparation
    loss = {"train":[], "val":[], "test":[]}
    best_mse = 9999999999
    PATH = "best_models" # Your PATH

    # Train the model
    print("Train starts!")
    start = time.time()

    for epoch in range(n_epochs):
        running_train_loss = 0.0
        running_val_loss = 0.0

        model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader)):  
            # origin shape: [32, 240, 320]
            # resized: [32, 1, 240, 320]
            images = images.unsqueeze(1).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            train_loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_train_loss +=train_loss.item()

        train_loss_value = running_train_loss/len(train_loader)
        loss["train"].append(train_loss_value)

        # Validation loop
        with torch.no_grad():
            model.eval()
            for images, labels in val_loader:
                images = images.unsqueeze(1).to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()

            # Calculate validation loss value 
            val_loss_value = running_val_loss/len(val_loader) 
            loss["val"].append(val_loss_value)
        
        # Save the model if MSE loss is the best
        if val_loss_value < best_mse:
            torch.save(model.state_dict(), os.path.join(PATH, f"CNN_ch{channels}_lr{lr}.pth"))
            best_mse = val_loss_value
            # print("new best model! Epoch:", epoch+1, " Lr:", lr)
    end = time.time()
    print("Run time [s]: ",end-start)

    return loss, model

# Test function
def test(loss, channels, lr, batch_size=32):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model that we saved at the end of the training loop
    model = CNN(channels).to(device)
    PATH = "best_models"
    path = os.path.join(PATH, f"CNN_ch{channels}_lr{lr}.pth")
    model.load_state_dict(torch.load(path)) 
    running_test_mse = 0.0

    print("Test starts!")
    with torch.no_grad(): 
        for images, labels in tqdm(test_loader):
            images = images.unsqueeze(1).to(device)
            labels = labels.to(device) 
      
            outputs = model(images)

            test_mse = F.mse_loss(outputs, labels)
            running_test_mse += test_mse.item()
            
        # Calculate test loss value 
        test_loss_value = running_test_mse/len(test_loader) 
        loss["test"].append(test_loss_value)
        print (f'Test MSE: {test_loss_value:.4f}')
    return loss

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test= prepare_data()
    
    # Parameters
    args = EasyDict({
    "batch_size": 32,
    "channels": [3,3],
    "lr": 0.1,
    "n_epochs": 30,
})
    
    # make loader
    train_loader = make_loader(X_train, y_train, args["batch_size"])
    val_loader = make_loader(X_val, y_val, args["batch_size"])
    test_loader = make_loader(X_test, y_test, args["batch_size"])

    print("Convolutional Neural Network")
    print("parameters:", args["channels"], args["lr"], args["batch_size"])
    
    # Train & Test
    loss, _ = train(**args)
    loss = test(loss, args["channels"], args["lr"], args["batch_size"])
    
    # Plot
#     plt.figure(figsize=(10,5))
#     plt.plot(loss["train"], label='Train MSE')
#     plt.plot(loss["val"], label='Validation MSE')

#     plt.title(f"CNN channels = {channels}")
#     plt.xlabel("Epoch")
#     plt.ylabel("MSE")
#     plt.legend()
#     plt.show()

    print("Test MSE", loss["test"])