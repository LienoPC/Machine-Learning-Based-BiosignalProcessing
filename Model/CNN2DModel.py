import torch.utils.data.dataset as dataset
from PIL import Image

import timm

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary
import torchvision.transforms as transforms

class SignalImageDataset(dataset.Dataset):
    """
    Dataset that elaborates spectrogram signal as images

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


class SpectrogramImageTransform():

    def __init__(self, resize_dim, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def transform(self, data):
        return self.transform(data)


def main_transfer_learning():

    ## Data loading
    img_size = 224 # We have 64x64 images but the network takes 224x224

    # Need to compute/read mean and std from the chosen dataset
    mean = (0.5,0.5,0.5)
    std = 0.1

    batch_size = 64

    path = "./Dataset"
    ## Define model characteristics

    classes = 2 # For now, we consider only two classes: stressed and not stressed

    # Choose and existant ImageNet-Trained model to work with
    model_name = 'resnet50'
    model = timm.create_model(model_name, pretrained = True, num_classes = classes)
    summary(model, (3,224,224))
    # We should evaluate to insert class rebalancing basing on the number of elements


    ## Define transfer learning criterion and learning rate scheduler
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    ## Create dataset
    img_list = ""
    label_list = ""
    # To correct with the right img and label objects
    train_set = SignalImageDataset(img_list, label_list)
    valid_set = SignalImageDataset(img_list, label_list)
    test_set = SignalImageDataset(img_list, label_list)