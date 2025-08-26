import csv
import math
import os
import pickle

import neurokit2
import numpy as np
import pandas as pd
import pywt
from matplotlib import pyplot as plt

from MLModels.FeatureExtraction import extract_features, extract_window_features_list, swt_denoise, separate_scl_scr, \
    swt_denoise_safe, normalize_signal, butter_bandpass_filter, butter_lowpass_filter
from Model.Dataset.WESAD import resample_labels, window_labels

WESAD_path = "../Model/Dataset/Data/WESAD/"
Data_path = "Data/"

def remove_invalid_labels_from_dataset():
    df = pd.read_csv(Data_path + "WESAD.csv", header=0)
    df = df[df["label"].isin([1, 2])].reset_index(drop=True)
    df["label"] = np.where(df["label"] == 1, 0, 1)
    print(f"Filtered dataset length: {len(df)}")
    df.to_csv(Data_path + "WESAD_filtered.csv", index=False, header=True)


def redefine_invalid_labels_from_dataset():
    df = pd.read_csv(Data_path + "WESAD.csv", header=0)

    df["label"] = pd.to_numeric(df["label"])

    df = df[df["label"] < 5].reset_index(drop=True)
    df = df[df["label"] > 0].reset_index(drop=True)
    df["label"] = np.where(df["label"] != 2, 0, 1)

    print(f"Redefined dataset length: {len(df)}")
    df.to_csv(Data_path + "WESAD_redefined.csv", index=False, header=True)


def build_dataset():
    dir_list = os.listdir(WESAD_path)
    feature_list = []
    labels_list = []
    subjects_list = []
    label_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}

    feature_names = ["MeanSCR", "MaxSCR", "MinSCR", "RangeSCR", "SkewenessSCR", "KurtosisSCR"]

    window_length = 15
    fs = 4
    overlap = 0
    window_size = int(window_length * fs)
    stride = int(window_size * (1 - overlap))

    eps = 1e-6
    subj_counter = 0
    for subj in dir_list:
        with open(os.path.join(WESAD_path, subj, f"{subj}.pkl"), "rb") as file:
            data = pickle.load(file, encoding="bytes")

            gsr_wrist = np.concatenate(data[b'signal'][b'wrist'][b'EDA'])

            total = len(gsr_wrist)
            n_epochs = int(np.floor((total - window_size) / stride) + 1)

            labels = np.array(data[b'label'])

            labels_resampled = resample_labels(labels, 700, 4, label_weights=label_weights)

            assert len(gsr_wrist) == len(labels_resampled)
            labels_window = window_labels(labels_resampled, window_size, stride, label_weights=label_weights)
            labels_list.extend(labels_window)
            #gsr_wrist = 100.0 / (gsr_wrist + eps)


            #gsr_wrist = normalize_signal(gsr_wrist)
            #gsr_wrist = butter_bandpass_filter(gsr_wrist, 0.00005, 1.9, fs, 1)

            '''
            gsr_wrist = butter_lowpass_filter(gsr_wrist, 0.5, fs)
            gsr_wrist = swt_denoise_safe(gsr_wrist, wavelet='db4', level=3)

            gsr_wrist = separate_scl_scr(gsr_wrist)

            '''

            features = extract_window_features_list(gsr_wrist, fs, window_size, stride, n_epochs)
            feature_list.extend(features)
            subjects_list.extend(np.full(len(features), subj_counter))
            assert len(feature_list) == len(labels_list)
            assert len(labels_list) == len(subjects_list)

        subj_counter += 1


    uniq_all, counts_all = np.unique(np.array(labels_list), return_counts=True)
    print("Accumulated labels_list distribution:")
    for u, c in zip(uniq_all, counts_all):
        print(f"  Label {u}: {c} occurrences")



    with open("Data/WESAD.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([*feature_names, "subject", "label"])
        for features, subject, label in zip(feature_list, subjects_list, labels_list):
            writer.writerow([*features, subject, label])


    print(f"Feature list length: {len(feature_list)}")
    print(f"Labels list length: {len(labels_list)}")
    print(f"Subjects list length: {len(subjects_list)}")
    # Verify if saved data is correct
    df = pd.read_csv(Data_path + "WESAD.csv", header=0)

    feature_list_saved = df[feature_names].values.tolist()
    label_list_saved = df["label"].astype(int).tolist()
    subjects_list_saved = df["subject"].astype(int).tolist()

    for a, b in zip(labels_list, label_list_saved):
        assert a == b
    for a, b in zip(subjects_list, subjects_list_saved):
        assert a == b


build_dataset()

remove_invalid_labels_from_dataset()
redefine_invalid_labels_from_dataset()