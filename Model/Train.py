import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.classification import BinaryF1Score

precision_recall = BinaryPrecisionRecallCurve()
f1_score = BinaryF1Score()
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, device='cpu'):
    """
    Support function for model training.

    :param model: Model to be trained
    :param criterion: Optimization criterion (loss)
    :param optimizer: Optimizer to use for training
    :param scheduler: Instance of ``torch.optim.lr_scheduler``
    :param dataloaders: Array of dataloaders in the form of ['train', 'val']
    :param dataset_sizes: Array of length for each phase dataset
    :param num_epochs: Number of epochs
    :param device: Device to run the training on. Must be 'cpu' or 'cuda'
    """
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    pr_curve = BinaryPrecisionRecallCurve(pos_label=1)
    f1_curve = []
    _, _, thresholds = pr_curve.compute()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
              model.train()  # Set model to training mode
            else:
              model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                  outputs = model(inputs)
                  _, preds = torch.max(outputs, 1) # TODO: Check again the logic of this line
                  loss = criterion(outputs, labels)

                  # Backward + optimize only if in training phase
                  if phase == 'train':
                    loss.backward()
                    optimizer.step()

                    # Compute precision-recall curve and f1-score
                    if phase == 'val':
                        pr_curve.update(outputs, labels)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
              scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            # Compute precision, recall and F1-score curve against thresholds for each epoch
            precision, recall, thresholds = pr_curve.compute()
            eps = 1e-8
            f1_curve = 2 * (precision * recall) / (precision + recall + eps)

            pr_curve.plot(score=True)
            pr_curve.reset()

            plot_f1score(f1_curve, thresholds)
            # Deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
              best_acc = epoch_acc
              best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    best_f1_idx = torch.argmax(f1_curve)
    best_f1 = f1_curve[best_f1_idx].item()
    best_threshold = thresholds[best_f1_idx].item()

    print(f"Best f1 score: {best_f1:4f} with threshold {best_threshold}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_threshold




def plot_f1score(f1_curve, thresholds):
    plt.plot(f1_curve, thresholds)
    plt.xlabel("F1-Score")
    plt.ylabel("Thresholds")
    plt.show()

def store_epoch_stats(accuracy, loss):
    os.open("/Stats/train_log.csv") # TODO: Store data from running training

