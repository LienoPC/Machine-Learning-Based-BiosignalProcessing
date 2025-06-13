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


# Array of all transfer learning modes to try with the associated learning rates
transfer_modes = [('whole', 0.001), ('last', 0.01), ('split', 0.0001, 0.01)]
models = ['resnet50', 'mobilenetv4_large_100', 'densenet121']
def main_transfer_learning():

    ## Data loading
    img_size = 224 # We have 64x64 images but the network takes 224x224

    # Need to compute/read mean and std from the chosen dataset
    mean = (0.5,0.5,0.5)
    std = 0.1

    ## Create dataloaders
    path = "./Dataset"
    img_list = ""
    label_list = ""
    # To correct with the right img and label objects
    train_set = SignalImageDataset(img_list, label_list)
    valid_set = SignalImageDataset(img_list, label_list)
    test_set = SignalImageDataset(img_list, label_list)


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

        Trai


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