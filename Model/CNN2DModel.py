import csv
import gc
from copy import deepcopy

import pandas as pd
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
import timm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary

from sklearn.model_selection import train_test_split

from Model.Dataset.SignalImageDataset import SignalImageDataset, ScalogramImageTransform
from Model.Train.Train import train_loop


# Array of all transfer learning modes to try with the associated learning rates
transfer_modes = [('whole', 0.0001), ('differential', (0.0001, 0.001))]
models = ['inception_resnet_v2']
def main_transfer_learning():

    ## Data loading
    img_size = 224 # The network takes 224x224x3

    mean = [0.1522, 0.4008, 0.7676]
    std = [0.2945, 0.3832, 0.2983]

    # Load dataset
    train_img_list, train_label_list = get_dataset_lists("./Dataset/train.csv")
    valid_img_list, valid_label_list = get_dataset_lists("./Dataset/valid.csv")


    # Transformation
    transform = ScalogramImageTransform(img_size, mean=mean, std=std)
    train_set = SignalImageDataset(train_img_list, train_label_list, transform.get_transform())
    valid_set = SignalImageDataset(valid_img_list, valid_label_list, transform.get_transform())

    # Batch size
    batch_size = 64

    # Compute sampler for class rebalancing
    train_sampler = get_weighted_random_sampler(train_label_list)
    valid_sampler = get_weighted_random_sampler(valid_label_list)

    # Dataloaders
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(dataset=valid_set, batch_size=batch_size, sampler=valid_sampler)

    for model in models:
        transfer_learn(model, train_dataloader, valid_dataloader)

def transfer_learn(model_name, train_loader, valid_loader):
    classes = 1

    ## Define model characteristics

    # Choose an existent ImageNet-Trained model to work with
    base_model = timm.create_model(model_name, pretrained=True, num_classes=classes)

    dataloaders = {'train': train_loader, 'valid': valid_loader}
    dataset_sizes = {'train': len(train_loader.dataset), 'valid': len(valid_loader.dataset)}

    for mode in transfer_modes:
        learning_rate = 0.001
        model = deepcopy(base_model)
        model.to('cuda')
        param_groups = [{'params': model.parameters()}]
        ## Define transfer learning criterion and learning rate scheduler
        if mode[0] in ('whole', 'last'):
            lr = mode[1]
            if mode[0] == 'last':
                model = freeze_non_fc_layers(model)
            param_groups = [{'params': model.parameters(), 'lr': lr}]

        elif mode[0] == 'differential':
            base_lr, head_lr = mode[1]
            base_params, head_params = [], []
            for n, p in model.named_parameters():
                if any(k in n for k in ('fc', 'classifier', 'head')):
                    head_params.append(p)
                else:
                    base_params.append(p)
            param_groups = [{'params': base_params, 'lr': base_lr},{'params': head_params, 'lr': head_lr}]

        #summary(model, (3, 224, 224))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(param_groups, lr=learning_rate)
        lr_decay = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        model, best_threshold = train_loop(model, f"{model_name}_{mode[0]}", criterion, optimizer, dataloaders, dataset_sizes, lr_decay=lr_decay, num_epochs=100)
        del model
        del optimizer
        del lr_decay
        torch.cuda.empty_cache()
        gc.collect()



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


def get_weighted_random_sampler(label_list):
    label_list = torch.tensor(label_list)
    class_sample_count = torch.tensor([(label_list == t).sum() for t in torch.unique(label_list)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[int(t)] for t in label_list])

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler

def subdivide_dataset(csv_path, csv_name, save_path):
    df = pd.read_csv(csv_path + csv_name, header=None, names=["img", "label"])

    train_val, test = train_test_split(df, test_size=0.10, stratify=df["label"], random_state=30)
    train, val = train_test_split(train_val, test_size=0.10/0.90, stratify=train_val["label"], random_state=30)
    train.to_csv(f"{save_path}/train.csv", index=False)
    val.to_csv(f"{save_path}/valid.csv", index=False)
    test.to_csv(f"{save_path}/test.csv", index=False)

def get_dataset_lists(csv_path):
    df = pd.read_csv(csv_path)  # header inferred from first row
    img_list = df["img"].tolist()
    label_list = df["label"].astype(int).tolist()
    return img_list, label_list


def create_dataset_():
    #subdivide_dataset("./Dataset/Data/WESAD/", "WESAD_filtered.csv", "./Dataset")

    train_img_list, train_label_list = get_dataset_lists("./Dataset/train.csv")
    valid_img_list, valid_label_list = get_dataset_lists("./Dataset/valid.csv")
    test_img_list, test_label_list = get_dataset_lists("./Dataset/test.csv")

    transform = ScalogramImageTransform(224)

    entire_img_list = train_img_list + valid_img_list + test_img_list
    entire_label_list = train_label_list + valid_label_list + test_label_list
    entire_set = SignalImageDataset(entire_img_list, entire_label_list, transform.get_transform())
    entire_dataloader = DataLoader(dataset=entire_set)

    mean, std = SignalImageDataset.compute_mean_std(entire_dataloader)

#subdivide_dataset('./Dataset/Data/', 'WESAD_filtered.csv', './Dataset')
#create_dataset_()
#main_transfer_learning()
