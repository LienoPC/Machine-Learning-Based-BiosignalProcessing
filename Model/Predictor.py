import timm
import torch
from torchvision import transforms
from PIL import Image

from Model.Dataset.SignalImageDataset import ScalogramImageTransform


class Predictor:
    def __init__(self, model_name, checkpoint_path, device='cuda', input_size = (224,224), mean=[0.0657, 0.2120, 0.7650], std=[0.1999, 0.3295, 0.2286], threshold = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Load the model and weights
        self.model = timm.create_model(model_name, pretrained=True, num_classes=1).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Set threshold for prediction
        if threshold:
            self.threshold = checkpoint['best_threshold']
        else:
            self.threshold = 0.5

        # Create data transform preprocess
        self.transform = ScalogramImageTransform(224, mean=mean, std=std).get_transform()

    @torch.no_grad()
    def predict(self, image):
        """
        Takes a scalogram image (as PIL image) and returns the predicted class value
        :param image:
        :return:
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x).view(-1)
        prob = torch.sigmoid(logits).item()
        label = int(prob > self.threshold)
        return prob, label
