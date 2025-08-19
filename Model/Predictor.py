import neurokit2
import numpy as np
import pandas as pd
import scipy
import timm
import torch
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
from PIL import Image
import joblib

from Model.Dataset.SignalImageDataset import ScalogramImageTransform

import abc
class Predictor:
    @abc.abstractmethod
    def predict(self, input):
        """
        Takes an input and returns the predicted class value
        :param image:
        :return:
        """
        return


class CNNPredictor(Predictor):
    def __init__(self, model_name, checkpoint_path, device='cuda', input_size = (224,224), mean=[0.1550, 0.4132, 0.7700], std=[0.2955, 0.3833, 0.2991], threshold = True):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Load the model and weights
        self.model = timm.create_model(model_name, pretrained=False, num_classes=1).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Set threshold for prediction
        if threshold:
            self.threshold = checkpoint['best_threshold']
            print(f"Best threshold loaded: {self.threshold}")
        else:
            self.threshold = 0.5

        # Create data transform preprocess
        self.transform = ScalogramImageTransform(224, mean=mean, std=std).get_transform()

        self.feature_names = ''


    @torch.no_grad()
    def predict(self, image):
        """
        Takes a scalogram image and returns the predicted class value
        :param image: scalogram image as PIL image
        :return:
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x).view(-1)
        prob = torch.sigmoid(logits).item()
        label = int(prob > self.threshold)
        torch.cuda.empty_cache()
        return prob, label


class MLPredictor(Predictor):

    def __init__(self, model_path, sampling_rate):
        self.models = []
        self.sampling_rate = sampling_rate
        self.model = self.load_model(model_path)

    def predict(self, signal):
        """
        Takes a 1D GSR signal, extract features and apply all available models
        :param signal: 1D array of signal to predict
        :param sampling_rate: sampling rate of the input signal
        :return:
        """
        X = self.extract_features(signal).reshape(1, -1)
        pipe = self.model
        label = self.model.predict(X)
        return label


    def extract_features(self, epoch_signal):

        pad_samples = len(epoch_signal)
        padded = np.pad(epoch_signal, (pad_samples, pad_samples), mode='reflect')
        # Extract SCR (Skin Conductance Response)
        scr_signal = neurokit2.eda_phasic(padded, self.sampling_rate)
        scr_signal = np.asarray(scr_signal["EDA_Phasic"])
        scr_signal = scr_signal[pad_samples: pad_samples + len(epoch_signal)]

        if len(scr_signal) == 0:
            return (0, 0, 0, 0, 0, 0)

        mean_scr = np.mean(scr_signal, axis=0)
        max_scr = np.max(scr_signal, axis=0)
        min_scr = np.min(scr_signal, axis=0)
        range_scr = max_scr - min_scr
        skeweness_scr = scipy.stats.skew(scr_signal)
        kurtosis_scr = scipy.stats.kurtosis(scr_signal)
        feat_array = np.asarray([mean_scr, max_scr, min_scr, range_scr, skeweness_scr, kurtosis_scr])


        return feat_array


    def load_model(self, joblib_file):

        artifact = joblib.load(joblib_file)
        model = artifact["model"]
        self.feature_names = artifact["feature_names"]

        return model