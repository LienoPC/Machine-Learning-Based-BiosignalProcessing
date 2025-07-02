import csv
import os
import cv2
import numpy as np
import timm
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC, BinaryPrecisionRecallCurve)
from Model.CNN2DModel import SignalImageDataset, get_dataset_lists
import seaborn as sns
import torchvision.transforms as transforms

from Model.Dataset.SignalImageDataset import ScalogramImageTransform

precision_recall = BinaryPrecisionRecallCurve()
f1_score = BinaryF1Score()


metrics_header = ['model_name', 'loss', 'accuracy', 'precision', 'recall', 'f1-score', 'roc_auc']

num_classes = 1
def test():
    model_name = "mobilenetv4_conv_large"
    model_path = "./Log/Saved/3/mobilenetv4_conv_large_differential/mobilenetv4_conv_large_differential_100.pt"

    init_test_csv(model_name)
    mean = [0.0657, 0.2120, 0.7650]
    std = [0.1999, 0.3295, 0.2286]

    transform = ScalogramImageTransform(224, mean=mean, std=std)

    batch_size = 64
    test_img_list, test_label_list = get_dataset_lists("./Dataset/test.csv")
    test_set = SignalImageDataset(test_img_list, test_label_list, transform.get_transform())
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    test_metrics = test_function(model_name, model_path, test_dataloader, criterion, thresold=True)
    store_test(test_metrics, model_name)


def test_function(model_name, model_path, test_loader, criterion, device="cuda", thresold=False):
    checkpoint = torch.load(model_path, map_location=device)
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    best_threshold = checkpoint['best_threshold']

    print("Best Threshold: ", best_threshold)

    # Define all metrics using the "best_threshold" computed during validation
    if thresold:
        acc_metric = BinaryAccuracy(threshold=best_threshold).to(device)
        prec_metric = BinaryPrecision(threshold=best_threshold).to(device)
        rec_metric = BinaryRecall(threshold=best_threshold).to(device)
        f1_metric = BinaryF1Score(threshold=best_threshold).to(device)
        cm_metric = ConfusionMatrix(num_classes=2, threshold=best_threshold, task='binary').to(device)
    else:
        acc_metric = BinaryAccuracy().to(device)
        prec_metric = BinaryPrecision().to(device)
        rec_metric = BinaryRecall().to(device)
        f1_metric = BinaryF1Score().to(device)
        cm_metric = ConfusionMatrix(num_classes=2,task='binary').to(device)

    auroc_metric = BinaryAUROC().to(device)
    pr_curve_metric = BinaryPrecisionRecallCurve().to(device)
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs).view(-1)

            # Update loss value
            loss = criterion(logits, labels.float())
            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits)
            # Update all metrics
            acc_metric.update(probs, labels)
            prec_metric.update(probs, labels)
            rec_metric.update(probs, labels)
            f1_metric.update(probs, labels)
            auroc_metric.update(probs, labels)
            pr_curve_metric.update(probs, labels)
            cm_metric.update(probs, labels)

    dataset_size = len(test_loader.dataset)
    total_loss = running_loss / dataset_size

    accuracy = acc_metric.compute().item()
    precision = prec_metric.compute().item()
    recall = rec_metric.compute().item()
    f1 = f1_metric.compute().item()
    auroc = auroc_metric.compute().item()
    precision_vals, recall_vals, thresholds = pr_curve_metric.compute()
    conf_mat = cm_metric.compute().cpu().numpy()

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
    :param metrics: object that contains all the metrics associated with the test loop
    :param model_name: name of the model tested
    :return:
    """
    save_dir = f"./Test/SavedTests"
    model_dir = f"{save_dir}/{model_name}"
    with open(f"{model_dir}/{model_name}.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Store data inside csv file
        writer.writerow([model_name, metrics['loss'], metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['roc_auc']])
        # Save pr_curve plot and confusion matrix as images
        store_pr_curve(metrics['pr_curve'], f"{model_dir}/pr_curve.png")
        store_conf_matrix(metrics['confusion_matrix'], f"{model_dir}/confusion_matrix.png")

def init_test_csv(model_name):
    """
    Function that initializes test csv file, so that it can be used for each test scenario
    :return:
    """
    save_dir = f"./Test/SavedTests"
    model_dir = f"{save_dir}/{model_name}"
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/{model_name}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(metrics_header)

def store_conf_matrix(confusion_matrix, path):
    plt.figure(figsize=(4, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def store_pr_curve(pr_curve, path):
    """
    Plots and saves PR curve plot as an image
    :param pr_curve: object saved in test metrics that returns precision_vals, recall_vals, thresholds
    :param path:
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

    f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    f1_vals_np = f1_vals.detach().cpu().numpy()
    best_idx = int(np.nanargmax(f1_vals_np))

    best_r, best_p, best_thresh = (recall_vals[best_idx], precision_vals[best_idx], thresholds[best_idx])
    plt.scatter(best_r.cpu(), best_p.cpu(), marker='o', label=f'Best F1 @ {best_thresh:.2f}')
    plt.legend()

    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


test()
