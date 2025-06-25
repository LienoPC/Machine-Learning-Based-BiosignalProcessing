import os

import cv2
import torch
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.classification import BinaryF1Score

from Model.Test.ConfusionMatrix import conf_matrix

precision_recall = BinaryPrecisionRecallCurve()
f1_score = BinaryF1Score()


test_loader = "testloader" #TODO: load here the test dataset

def test_function(model_name, model_path, test_loader, dataset_size, criterion, device='cuda'):
    model = torch.load(model_path)
    # Build confusion matrix
    conf_image = conf_matrix(test_loader, model, 1)

    _, _, thresholds = precision_recall.compute()

    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = (outputs >= 0.5).long()
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / dataset_size
    total_acc = float(running_corrects) / dataset_size

    store_test(total_loss, total_acc, model_name, conf_image)

def store_test(total_loss, total_acc, model_name, conf_image):
    """
    Function that saves all data associated with a test loop
    :param list_err_file: TODO : Change this accordingly to data saved
    :param model_name: name of the model tested
    :return:
    """
    save_dir = f"./SavedTests/{model_name}/"
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    output_file = open(f"./SavedTests/{model_name}/Test.txt", "w")
    output_file.writelines([f"Test Loss: {total_loss}", f"Test Accuracy: {total_acc}"])

    cv2.imwrite(f"./SavedTests/{model_name}/Test.png", conf_image)