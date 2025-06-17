import cv2
import numpy as np
import torch
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from PIL import Image

import timm

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary
import torchvision.transforms as transforms

class SignalImageDataset(dataset.Dataset):
    """
    Dataset that elaborates scalogram images

    img_list: list of images path
    label_list: path of the list of labels
    """

    def __init__(self, img_list, label_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        # Transform operations to be applied on input images
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list['item']
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = self.label_list['item']

        return image, label


class ScalogramImageTransform():

    def __init__(self, resize_dim, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def transform(self, data):
        return self.transform(data)


def compute_mean_std(dataloader, size):
    psum = torch.tensor([0.0]) # Pixel sum
    psum_sq = torch.tensor([0.0]) # Squared sum

    for image in dataloader:
        psum += image.sum(axis=[0, 2, 3])
        psum_sq += (image**2).sum(axis=[0, 2, 3])

    # Pixel count
    count = len(dataloader)*size

    total_mean = psum/count
    total_var = (psum_sq/count) - (total_mean**2)
    total_std = torch.sqrt(total_var)
    print('- mean: {:.4f}'.format(total_mean.item()))
    print('- std:  {:.4f}'.format(total_std.item()))
    return total_mean, total_std


# Array of all transfer learning modes to try with the associated learning rates
transfer_modes = [('whole', 0.001), ('last', 0.01), ('split', 0.0001, 0.01)]
models = ['resnet50', 'mobilenetv4_large_100', 'densenet121']
def main_transfer_learning():

    ## Data loading
    img_size = 224 # We have ??x?? images but the network takes 224x224x3

    # TODO: Need to compute/read mean and std from the chosen dataset
    mean = 0
    std = 1

    # Load dataset
    path = "./Dataset"
    train_img_list = ""
    train_label_list = ""

    test_img_list = ""
    test_label_list = ""

    # Transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=mean, std=std)
    ])
    # TODO: change with the right img and label objects
    train_set = SignalImageDataset(train_img_list, train_label_list, transform)
    test_set = SignalImageDataset(test_img_list, test_label_list, transform)

    # Batch size
    batch_size = 128

    # Create data split and loaders
    split_ratio = 0.2
    train_size = len(train_set)
    dataset_indices = list(range(train_size))
    split_index = int(np.floor(split_ratio*train_size))

    train_idx, val_idx = dataset_indices[split_index:], dataset_indices[:split_index]
    train_sampler, val_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)

    # Dataloaders
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=val_sampler)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

def transfer_learn(model_name, train_loader, valid_loader, test_loader):
    classes = 2 # For now, we consider only two classes: stressed and not stressed
    batch_size = 64

    ## Define model characteristics

    # Choose and existant ImageNet-Trained model to work with
    model = timm.create_model(model_name, pretrained=True, num_classes=classes)
    summary(model, (3, 224, 224))
    # We should evaluate to insert class rebalancing basing on the number of elements



    for mode in transfer_modes:
        learning_rate = 0.001
        if mode[0] == 'whole':
            learning_rate = mode[1]
        if mode[0] == 'last':
            learning_rate = mode[1]
            model = freeze_non_fc_layers(model)
        ## Define transfer learning criterion and learning rate scheduler
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        lr_decay = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)




def freeze_non_fc_layers(model):
    """
        Freezes all parameters except those in fully‑connected (Linear) layers.
    """
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = True

    return model