import csv
import os
import cv2
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC, BinaryPrecisionRecallCurve)
from Model.CNN2DModel import SignalImageDataset
from Model.Test.ConfusionMatrix import conf_matrix
import torchvision.transforms as transforms

precision_recall = BinaryPrecisionRecallCurve()
f1_score = BinaryF1Score()

test_results_csv = "test_results.csv"
metrics_header = ['model_name', 'loss', 'accuracy', 'precision', 'recall', 'f1-score', 'roc_auc']


def test():
    model_name = "resnet50"  # TODO: set here the correct name
    model_path = "/Log/Train/"  # TODO: set here the correct path

    # TODO: Need to compute/read mean and std from the chosen dataset
    mean = 0
    std = 1

    transform = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=mean, std=std)
    ])

    # TODO: Change with real values from test set
    batch_size = 64
    test_img_list = ""
    test_label_list = ""
    test_set = SignalImageDataset(test_img_list, test_label_list, transform)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    test_metrics = test_function(model_name, model_path, test_dataloader, )
    store_test(test_metrics, model_name)

    criterion = nn.BCELoss()

    test_metrics = test_function(model_name, model_path, test_dataloader, criterion)
    store_test(test_metrics, model_name)


def test_function(model_name, model_path, test_loader, criterion):
    model = torch.load(model_path)
    best_treshold = model.best_treshold
    # Build confusion matrix
    conf_image = conf_matrix(test_loader, model, 1)

    # Define all metrics using the "best_threshold" computed during validation
    acc_metric = BinaryAccuracy(threshold=best_treshold)
    prec_metric = BinaryPrecision(threshold=best_treshold)
    rec_metric = BinaryRecall(threshold=best_treshold)
    f1_metric = BinaryF1Score(threshold=best_treshold)
    auroc_metric = BinaryAUROC()
    pr_curve_metric = BinaryPrecisionRecallCurve()

    cm_metric = ConfusionMatrix(num_classes=2, threshold=best_treshold, task='binary')

    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).view(-1)

            # Update loss value
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # Update all metrics
            acc_metric.update(outputs, labels)
            prec_metric.update(outputs, labels)
            rec_metric.update(outputs, labels)
            f1_metric.update(outputs, labels)
            auroc_metric.update(outputs, labels)
            pr_curve_metric.update(outputs, labels)
            cm_metric.update(outputs, labels)

    dataset_size = len(test_loader.dataset)
    total_loss = running_loss / dataset_size

    accuracy = acc_metric.compute().item()
    precision = prec_metric.compute().item()
    recall = rec_metric.compute().item()
    f1 = f1_metric.compute().item()
    auroc = auroc_metric.compute().item()
    precision_vals, recall_vals, thresholds = pr_curve_metric.compute()
    conf_mat = cm_metric.cpu().numpy()

    test_metrics = {
        'loss': total_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auroc,
        'confusion_matrix': conf_mat,
        'pr_curve': (precision_vals, recall_vals, thresholds)
    }
    return test_metrics

def store_test(metrics, model_name):
    """
    Function that saves all data associated with a test loop
    :param list_err_file: TODO : Change this accordingly to data saved
    :param model_name: name of the model tested
    :return:
    """
    save_dir = f"./SavedTests/"

    with open(os.path.join(save_dir, test_results_csv), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Store data inside csv file
        writer.writerow([model_name, metrics['loss'], metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['roc_auc']])
        # Save pr_curve plot and confusion matrix as images
        store_pr_curve(metrics['pr_curve'], os.path.join(f"./SavedTests/{model_name}/", "pr_curve.png"))
        cv2.imwrite(f"./SavedTests/{model_name}/conf_matrix.png", metrics['confusion_matrix'])

def init_test_csv():
    """
    Function that initializes test csv file, so that it can be used for each test scenario
    :return:
    """
    save_dir = f"./SavedTests/"
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, test_results_csv), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(metrics_header)

def store_pr_curve(pr_curve, path):
    """
    Plots and saves PR curve plot as an image
    :param pr_curve: object saved in test metrics that returns precision_vals, recall_vals, thresholds
    :param path: file path to save PR curve
    :return:
    """
    precision_vals, recall_vals, thresholds = pr_curve
    plt.figure(figsize=(6, 6))
    plt.step(recall_vals.cpu().numpy(), precision_vals.cpu().numpy(), where='post')
    plt.fill_between(
        recall_vals.cpu().numpy(),
        precision_vals.cpu().numpy(),
        step='post',
        alpha=0.2
    )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision–Recall curve (AUC = {torch.trapz(precision_vals, recall_vals):.3f})')

    best_idx = torch.nanargmax(2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8))
    best_r, best_p, best_thresh = (recall_vals[best_idx], precision_vals[best_idx], thresholds[best_idx])
    plt.scatter(best_r.cpu(), best_p.cpu(), marker='o', label=f'Best F1 @ {best_thresh:.2f}')
    plt.legend()

    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

init_test_csv()