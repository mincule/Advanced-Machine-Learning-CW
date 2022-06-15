from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load Data
def prepare_data(n_train=3000, n_val=150, n_test=300):
    img_train = []
    img_val = []
    img_test = []

    for i in tqdm(range(1,n_train+1)):
        ImagePATH = "data/Training/depth/"
        img_name = "depth_1_" + str(i).zfill(7) + ".png"
        ImagePATH = ImagePATH + img_name
        img_train.append(np.array(Image.open(ImagePATH), dtype=np.float32))

    for i in tqdm(range(n_train+1,n_train+n_val+1)):
        ImagePATH = "data/Training/depth/"
        img_name = "depth_1_" + str(i).zfill(7) + ".png"
        ImagePATH = ImagePATH + img_name
        img_val.append(np.array(Image.open(ImagePATH), dtype=np.float32))

    for i in tqdm(range(1,n_test+1)):
        ImagePATH = "data/Testing/depth/"
        img_name = "depth_1_" + str(i).zfill(7) + ".png"
        ImagePATH = ImagePATH + img_name
        img_test.append(np.array(Image.open(ImagePATH), dtype=np.float32))

    X_train = np.array(img_train, dtype=np.float32)
    X_val = np.array(img_val, dtype=np.float32)
    X_test = np.array(img_test, dtype=np.float32)

    # Mean and std of Depth map
    mean=1881.42
    std=12.29

    # Standardise data
    X_train -= mean
    X_train /= std
    X_val -= mean
    X_val /= std
    X_test -= mean
    X_test /= std

    print("**Experiment Dataset**")
    print("train image dim:", X_train.shape)
    print("val image dim:", X_val.shape)
    print("test image dim:", X_test.shape)

    whole_y_train = pd.read_csv("data/Training/Annotation_Training.csv",
                                usecols = [i for i in range(1,64)],
                                skiprows = [1,2,3])
    whole_y_test = pd.read_csv("data/Testing/Annotation_Testing.csv",
                               usecols = [i for i in range(1,64)],
                               skiprows = [1,2,3])
    whole_y_train.set_axis([i for i in range(63)], axis=1, inplace=True)
    whole_y_test.set_axis([i for i in range(63)], axis=1, inplace=True)

    y_train = whole_y_train[:n_train]
    y_val = whole_y_train[n_train:n_train+n_val]
    y_test = whole_y_test[:n_test]
    return X_train, y_train, X_val, y_val, X_test, y_test

# Dataloader
def make_loader(X, y, bs):
    """X, y: ndarray"""
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    return loader

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test= prepare_data(8,2,4)
    
    train_loader = make_loader(X_train, y_train, 8)
    print(train_loader)
    
    for img, labels in train_loader:
        print(img)
        print(labels)