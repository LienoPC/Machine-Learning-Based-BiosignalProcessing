import copy
import matplotlib.pyplot as plt
import time
import torch
import csv
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.classification import BinaryF1Score

precision_recall = BinaryPrecisionRecallCurve()
f1_score = BinaryF1Score()

model_file_path = "/Log/Train/"

def train_loop(model, model_name, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25, device='cuda', lr_decay=None):
    """
    Support function for model training.

    :param model: Model to be trained
    :param criterion: Optimization criterion (loss)
    :param optimizer: Optimizer to use for training
    :param scheduler: Scheduler for optimization
    :param dataloaders: Array of dataloaders in the form of ['train', 'val']
    :param dataset_sizes: Array of length for each phase dataset
    :param num_epochs: Number of epochs
    :param device: Device to run the training on. Must be 'cpu' or 'cuda'
    """
    start_time = time.time() # Timestamp for training time

    initialize_model_stats(model_name)

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
                  preds = (outputs >= 0.5).long() # TODO: Supposing to have already sigmoid on the model, change in case
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

            if phase == 'train' and lr_decay:
                lr_decay.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print(f"{model_name} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            store_epoch_stats(model_name, epoch_acc, epoch_loss, epoch, phase)
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
    print(f"{model_name} training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"{model_name} best val Acc: {best_acc:4f}")

    best_f1_idx = torch.argmax(f1_curve)
    best_f1 = f1_curve[best_f1_idx].item()
    best_threshold = thresholds[best_f1_idx].item()
    store_f1_curve(model_name, best_f1)
    print(f"Best f1 score: {best_f1:4f} with threshold {best_threshold}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Save then the model with inside the best_threshold for future usage
    store_model_checkpoint(model, optimizer, best_threshold, num_epochs, model_name)
    return model, best_threshold




def plot_f1score(f1_curve, thresholds):
    '''
    Plots computed f1_curve
    '''
    plt.plot(f1_curve, thresholds)
    plt.xlabel("F1-Score")
    plt.ylabel("Thresholds")
    plt.show()

def initialize_model_stats(model_name):
    '''
    Initialization of CSV file on which all training stats will be saved
    :param model_name: Name of the model to save
    '''
    with open(f'{model_file_path}/{model_name}/{model_name}.csv', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Phase', 'Accuracy', 'Loss'])

def store_epoch_stats(model_name, accuracy, loss, epoch, phase):
    '''
    Saves a single epoch data into training stats
    :param model_name: Name of the model to save
    :param accuracy: Epoch accuracy value
    :param loss: Epoch loss value
    :param epoch: Epoch number
    :param phase: Phase 'train' or 'val'
    :return:
    '''
    with open(f'/Log/Train/{model_name}.csv', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, phase, accuracy, loss])

def store_f1_curve(model_name, f1_curve):
    '''
    Saves values of f1 curve for plotting at the end of the training
    :param model_name: Name of the model to save
    :param f1_curve: F1 Curve
    :return:
    '''
    with open(f'/Log/Train/{model_name}.csv', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([['F1_curve'], f1_curve])

def store_model_checkpoint(model, optimizer, best_threshold, epoch, model_name):
    '''
    Save weights of the trained model
    :param model:
    :return:
    '''
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_threshold': best_threshold,
        'epoch': epoch,
    }
    torch.save(checkpoint, f"{model_file_path}/{model_name}/{model_name}_{epoch}.pt")

