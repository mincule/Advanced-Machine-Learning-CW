from PIL import Image

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

# Model: 3-layers Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=63, dropout=nn.Dropout(0.5)):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        x = self.linear(x)
        return x

# Train function
def train(batch_size, mlp_hidden_size, lr, n_epochs):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = MLP(76800, mlp_hidden_size).to(device)
    
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
            # resized: [32, 76800]
            images = images.reshape(batch_size, -1).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            outputs = outputs.squeeze(1)
            train_loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_train_loss +=train_loss.item()

        train_loss_value = running_train_loss/len(train_loader)
        loss["train"].append(train_loss_value)

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss_value:.4f}')

        # Validation loop
        with torch.no_grad():
            model.eval()
            for images, labels in val_loader:
                images = images.reshape(batch_size, 1, -1).to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                outputs = outputs.squeeze(1)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()

            # Calculate validation loss value 
            val_loss_value = running_val_loss/len(val_loader) 
            loss["val"].append(val_loss_value)
            if (epoch+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Validation Loss: {val_loss_value:.4f}')
        
        # Save the model if MSE loss is the best
        if val_loss_value < best_mse:
            torch.save(model.state_dict(), os.path.join(PATH, f"MLP_hs{mlp_hidden_size}_lr{lr}.pth"))
            best_mse = val_loss_value
            # print("new best model! Epoch:", epoch+1, " Lr:", lr)
    end = time.time()
    print("Run time [s]: ",end-start)
    return loss, model

# Test function
def test(loss, mlp_hidden_size, lr, batch_size=32):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model that we saved at the end of the training loop
    model = MLP(76800, mlp_hidden_size).to(device)
    PATH = "best_models"
    path = os.path.join(PATH, f"MLP_hs{mlp_hidden_size}_lr{lr}.pth")
    model.load_state_dict(torch.load(path)) 
    running_test_mse = 0.0

    print("Test starts!")
    with torch.no_grad(): 
        for images, labels in tqdm(test_loader):
            images = images.reshape(batch_size, -1).to(device)
            labels = labels.to(device) 
      
            outputs = model(images)
            outputs = outputs.squeeze(1)

            test_mse = F.mse_loss(outputs, labels)
            running_test_mse += test_mse.item()
            
        # Calculate test loss value 
        test_loss_value = running_test_mse/len(test_loader) 
        loss["test"].append(test_loss_value)
        print (f'Test MSE: {test_loss_value:.4f}')
    return loss

if __name__ == "__main__":
   # Load data
    try:
        X_train, y_train, X_val, y_val, X_test, y_test= prepare_data()
    except:
        X_train = np.load("data/X_train.npy")
        X_val = np.load("data/X_val.npy")
        X_test = np.load("data/X_test.npy")

        # y labels
        whole_y_train = pd.read_csv("data/Training/Annotation_Training.csv",
                                    usecols = [i for i in range(1,64)],
                                    skiprows = [1,2,3])
        whole_y_test = pd.read_csv("data/Testing/Annotation_Testing.csv",
                              usecols = [i for i in range(1,64)],
                              skiprows = [1,2,3])

        whole_y_train.set_axis([i for i in range(63)], axis=1, inplace=True)
        whole_y_test.set_axis([i for i in range(63)], axis=1, inplace=True)

        y_train = whole_y_train[:3000]
        y_val = whole_y_train[3000:3150]
        y_test = whole_y_test[:300]

#         X_train = X_train[:4]
#         X_val = X_val[:4]
#         X_test = X_test[:4]
#         y_train = y_train[:4]
#         y_val = y_val[:4]
#         y_test = y_test[:4]
    
    # Parameters
    args = EasyDict({
    "batch_size": 32,
    "mlp_hidden_size": 77,
    "lr": 0.1,
    "n_epochs": 30,
    })
    
    # make loader
    train_loader = make_loader(X_train, y_train, args["batch_size"])
    val_loader = make_loader(X_val, y_val, args["batch_size"])
    test_loader = make_loader(X_test, y_test, args["batch_size"])

    print("Multi-Layer Percetron")
    print("parameters:", args["mlp_hidden_size"], args["lr"], args["batch_size"])
    
    # Train & Test
    loss, _ = train(**args)
    loss = test(loss, args["mlp_hidden_size"], args["lr"], args["batch_size"])
    
    # Plot
#     plt.figure(figsize=(10,5))
#     plt.plot(loss["train"], label='Train MSE')
#     plt.plot(loss["val"], label='Validation MSE')

#     plt.title(f"MLP hs = {mlp_hidden_size}")
#     plt.xlabel("Epoch")
#     plt.ylabel("MSE")
#     plt.legend()
#     plt.show()

    print("Test MSE", loss["test"])
