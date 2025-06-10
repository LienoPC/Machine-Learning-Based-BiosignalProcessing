from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.classification import BinaryF1Score

precision_recall = BinaryPrecisionRecallCurve()
f1_score = BinaryF1Score()

testloader = "testloader" #TODO: load here the test dataset

def test_function(model, testloader):
    print("Work in progress...")