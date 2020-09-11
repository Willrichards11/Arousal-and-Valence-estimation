from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import pandas as pd
import os
import numpy as np
import torchvision.transforms.functional as F
from numpy import random
import argparse


class SubsetAdapted(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, additional, labels = self.dataset[self.indices[idx]]
        if self.transform:
            im = self.transform(im.astype(np.double))

        return im.float(), additional, labels

    def __len__(self):
        return len(self.indices)


class AmigosDatasetAdapted(Dataset):
    r"""
    lazy loading dataset for the adapted model (either hog or qlzm )

    Arguments:
        csv (Dataset): The model targets
        root_dir (sequence): data directory root
        hog : boolean - true = hog, false = qlzm
        reduce dataset: boolean - true = reduced, false = original
    """
    def __init__(self, csv, root_dir, hog, reduced_dataset=False):
        self.labels = pd.read_csv(
            csv, index_col=[0,1,6], converters={'targets': eval}
        )
        self.root_dir = root_dir

        if reduced_dataset == True:
            self.image_dir = os.path.join(self.root_dir, 'ImagesOversamplesecondattempt')
        else:
            self.image_dir = os.path.join(self.root_dir, 'Images')

        self.hog = hog
        if self.hog == True:

            self.adapted_dir = os.path.join(root_dir, "hog")
            self.map = {idx: file for idx, file in enumerate(os.listdir(self.image_dir))}
            self.len = len(self.map.keys())
        else:
            self.adapted_dir = os.path.join(root_dir, "output")
            self.map = {idx: file for idx, file in enumerate(os.listdir(self.image_dir))}
            self.len = len(self.map.keys())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        '''
            get an item by its index
        '''
        file_name = self.map[idx]
        img_name = os.path.join(self.image_dir, file_name)

        image = cv2.imread(img_name)

        i1, i2, i3 = [int(i) for i in file_name.split(",")[:3]]
        arous = self.labels.loc[(i1, str(i2), i3), 'arousal']
        valen = self.labels.loc[(i1, str(i2), i3), 'valence']

        if self.hog == True:
            additional_name = file_name.split(".")[0] + '.npy'
            additional_feature = np.load(os.path.join(self.adapted_dir, additional_name))
        else:
            additional_name = file_name.split(".")[0]
            additional_feature = np.loadtxt(os.path.join(self.adapted_dir, additional_name))

        return (image, additional_feature, np.array([arous, valen]) )


class AmigosDataset(Dataset):
   r"""
    lazy loading dataset for the non adapted model
    Arguments:
        csv (Dataset): The model targets
        root_dir (sequence): data directory root
        reduce dataset: boolean - true = reduced, false = original
    """
    def __init__(self, csv, root_dir, reduced_dataset):
        self.labels = pd.read_csv(
            csv, index_col=[0,1,6], converters={'targets': eval}
        )
        self.root_dir = root_dir
        if reduced_dataset:
            print ('using reduced dataset')
            self.image_dir = os.path.join(root_dir, 'ImagesOversamplesecondattempt')
        else:
            print ('Not using reduced dataset')

            self.image_dir = os.path.join(root_dir, 'Images')

        self.map = {idx:file for idx, file in enumerate(os.listdir(self.image_dir))}
        self.len = len(self.map.keys())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_name = self.map[idx]
        img_name = os.path.join(
            self.image_dir,
            file_name
        )
        image = cv2.imread(img_name)
        i1, i2, i3 = [int(i) for i in file_name.split(",")[:3]]

        arous = self.labels.loc[(i1, str(i2), i3), 'arousal']
        valen = self.labels.loc[(i1, str(i2), i3), 'valence']

        return (image, np.array([arous, valen]))


class AmigosDatasetInterSubject(Dataset):
   r"""
    lazy loading dataset for the inter subject validation  model

    Arguments:
        csv (Dataset): The model targets
        root_dir (sequence): data directory root
        subject : int - particpant to leave out
        reduce dataset: boolean - true = reduced, false = original
    """
    def __init__(self, csv, root_dir, subject, reduced_dataset):
        self.labels = pd.read_csv(
            csv, index_col=[0,1,6], converters={'targets': eval}
        )
        self.root_dir = root_dir

        if reduced_dataset:
            self.image_dir = os.path.join(root_dir, 'ImagesOversamplesecondattempt')
        else:
            self.image_dir = os.path.join(root_dir, 'Images')

        self.map = {
            idx : file for idx, file in enumerate(os.listdir(self.image_dir)) if
            int(file.split(",")[0]) == subject
        }
        self.len = len(self.map.keys())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_name = self.map[idx]
        img_name = os.path.join(
            self.image_dir,
            file_name
        )
        image = cv2.imread(img_name)
        i1, i2, i3 = [int(i) for i in file_name.split(",")[:3]]
        arous = self.labels.loc[(i1, str(i2), i3), 'arousal']
        valen = self.labels.loc[(i1, str(i2), i3), 'valence']

        return (image, np.array([arous, valen]))


class AmigosDatasetInterSubjectAdapted(Dataset):
   r"""
    lazy loading dataset for the adapted inter subject validation  model

    Arguments:
        csv (Dataset): The model targets
        root_dir (sequence): data directory root
        subject : int - particpant to leave out
        hog : boolean - true = hog, false = qlzm
        reduce dataset: boolean - true = reduced, false = original
    """
    def __init__(self, csv, root_dir, subject, hog, reduced_dataset):
        self.labels = pd.read_csv(
            csv, index_col=[0,1,6], converters={'targets': eval}
        )
        self.root_dir = root_dir
        self.hog = hog
        if reduced_dataset == True:
            self.image_dir = os.path.join(root_dir, 'ImagesOversamplesecondattempt')

        else:
            self.image_dir = os.path.join(root_dir, 'Images')
        self.map = {idx: file for idx, file in enumerate(os.listdir(self.image_dir))}

        if self.hog == True:
            self.adapted_dir = os.path.join(root_dir, "hog")
            self.len = len(self.map.keys())
        else:
            self.adapted_dir = os.path.join(root_dir, "output")
            self.len = len(self.map.keys())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_name = self.map[idx]
        img_name = os.path.join(
            self.image_dir,
            file_name
        )

        image = cv2.imread(img_name)
        i1, i2, i3 = [int(i) for i in file_name.split(",")[:3]]
        arous = self.labels.loc[(i1, str(i2), i3), 'arousal']
        valen = self.labels.loc[(i1, str(i2), i3), 'valence']


        if self.hog == True:
            addit_name = file_name.split(".")[0] + ".npy"
            additional_feature = np.load(os.path.join(self.adapted_dir, addit_name))
        else:
            addit_name = file_name.split(".")[0]
            additional_feature = np.loadtxt(os.path.join(self.adapted_dir, addit_name))
        return (image, additional_feature, np.array([arous, valen]) )



class Subset(Dataset):
    r"""
    custom subset of a dataset given specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform: The transforms to apply to images
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        if self.transform:
            im = self.transform(im.astype(np.double))

        return im.float(), labels

    def __len__(self):
        return len(self.indices)


def run_epoch(net, loader, optimizer, criterion):
    r"""
    Runs an entire epoch using a given data loader, optimizer and criterion

    Arguments:
    net : the neural net to use for the epoch
    loader: a dataloader to load each batch
    optimizer: stochastic gradient descent or adam
    criterion: Mean squared error for this project
    """
    net.train()
    epoch_loss = np.array([], dtype=np.float32)
    for i, data in enumerate(loader, 1):
        images = data[0].cuda().float()
        labels = data[-1].cuda().float()

        optimizer.zero_grad()
        if len(data) == 3:
            additional = data[1].cuda().float()
            outputs = net(images, additional)
        else:
            outputs = net(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss = np.append(epoch_loss, loss.data.item())
        # print (f"batch {i} | running_loss{np.mean(epoch_loss):.4f}")

    return net


def eval_model(net, loader, criterion):
    r"""
    Evaluates model performance
    Arguments:
    net : the neural net to use for the epoch
    loader: a dataloader to load each batch
    optimizer: stochastic gradient descent or adam
    criterion: Mean squared error for this project
    """
    net.eval()
    epoch_loss = np.array([], dtype=np.float32)
    print ('evaluating validation set')

    for i, data in enumerate(loader, 1):
        images = data[0].cuda().float()
        labels = data[-1].cuda().float()

        if len(data) == 3:
            additional = data[1].cuda().float()
            outputs = net(images, additional)
        else:
            outputs = net(images)

        loss = criterion(outputs, labels)
        epoch_loss = np.append(epoch_loss, loss.data.item())
    net.train()
    return np.mean(epoch_loss)


class RandomMirror(object):
    """ randomly mirror an image with 50% probability"""
    def __init__(self):
        pass

    def __call__(self, image):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]

        return image.copy()
