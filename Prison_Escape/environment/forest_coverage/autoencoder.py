""" This file is used to compress the state space of the TL coverarge. Currently only does the forest cover from a single csv file"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
# from simulator.forest_coverage.convert_multiple_maps import read_map_file
import cv2
import seaborn as sns

# from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
# from osgeo import gdal

import matplotlib.pyplot as plt
import pandas as pd

class TLCoverageDataset(Dataset):
    """ This function is used to create a dataset of the forest coverages.
        :param: dataset_dir: the directory of the forest coverages
        :param: store_memory: if true, store all the forest coverages in memory. If false, read the forest coverages from disk each time."""
    def __init__(self, dataset_dir, store_memory=True):
        self.dataset_dir = dataset_dir
        self.store_memory = store_memory
        
        self.file_paths = glob.glob(self.dataset_dir + '/*.npy', recursive=True)
        
        # read all into memory
        self.files = []
        if store_memory:
            for file_path in self.file_paths:
                map = cv2.resize(np.load(file_path), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                self.files.append(np.expand_dims(map, 0))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.store_memory:
            return self.files[idx]
        else:
            map_path = self.file_paths[idx]
            # map_array = np.expand_dim(read_map_file(map_path), 0)
            map_array = cv2.resize(np.load(map_path), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            return np.expand_dims(map_array, 0)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(1, 1)

        # # Decoder
        self.t_conv1 = nn.ConvTranspose2d(1, 4, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(4, 16, kernel_size=2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 1,kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # deconvolutions
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = torch.sigmoid(self.t_conv3(x))
        return x

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        return x

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

def train(forest_coverage_dataset_path, n_epochs, batch_size, learning_rate, save_dir=None):
    #Instantiate the model
    model = ConvAutoencoder()

    #Loss function
    criterion = nn.BCELoss()

    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = get_device()
    # print(device)
    model.to(device)

    #Get the training and test data
    tl_dataset = TLCoverageDataset(dataset_dir=forest_coverage_dataset_path)
    train_loader = torch.utils.data.DataLoader(tl_dataset, batch_size=batch_size, num_workers=0)
    # test_loader = torch.utils.data.DataLoader(tl, batch_size=32, num_workers=0)
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0

        #Training
        for map_tensor in train_loader:
            # print(type(map_arrays))
            # map_tensor = torch.tensor(map_arrays).float().to(device)
            map_tensor = map_tensor.float().to(device)
            # print(map_tensor.shape)
            optimizer.zero_grad()
            outputs = model(map_tensor)
            loss = criterion(outputs, map_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*map_tensor.size(0)
            
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    return model

def test(model):
    # model = torch.load(model_dir)
    # #Batch of test images

    map_input = np.load("simulator/forest_coverage/maps/0.npy")
    map_input = cv2.resize(map_input, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    # print(np.max(map_input), np.min(map_input))

    sns.heatmap(map_input)
    plt.savefig("input.png")

    # plt.savefig("output.png")
    map_array = torch.tensor(map_input).float().to("cuda")
    map_array = map_array.view(1, 1, 256, 256)
    
    # print(map_array.shape)

    print(model.encoder(map_array).shape)

    #Sample outputs
    output = model(map_array)
    # print(output.shape)
    image = output.detach().cpu().numpy()

    plt.figure()
    sns.heatmap(image[0,0,:,:])
    plt.savefig("output.png")

    map_resized = cv2.resize(map_input, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    plt.figure()
    sns.heatmap(map_resized)
    plt.savefig("output_resize.png")

    # print(((map_input - image[0])**2).mean(axis=None))
    # print(np.square(np.subtract(map_input[0], image[0,0])).mean(axis=None))

def produce_terrain_embedding(model: ConvAutoencoder, terrain_array: np.ndarray) -> np.ndarray:
    terrain_size = 256
    terrain_resized = cv2.resize(terrain_array, (terrain_size, terrain_size), interpolation=cv2.INTER_CUBIC)
    
    terrain_tensor = torch.tensor(terrain_resized).float().view(1, 1, terrain_size, terrain_size)
    embedding = model.encoder(terrain_tensor)
    
    return embedding.flatten().detach().cpu().numpy()

if __name__ == "__main__":
    # forest_coverage_dataset_path = "simulator/forest_coverage/maps"
    # model = train(forest_coverage_dataset_path, n_epochs=2000, batch_size=32, learning_rate=0.001)
    # torch.save(model, "autoencoder.pt")
    # torch.save(model.state_dict(), "autoencoder.pt")

    # model = ConvAutoencoder()
    # model.load_state_dict(torch.load('/nethome/sye40/PrisonerEscape/autoencoder.pt'))
    model = torch.load("simulator/forest_coverage/autoencoder.pt")

    torch.save(model.state_dict(), "autoencoder_state_dict.pt")
    

    # test(model)

