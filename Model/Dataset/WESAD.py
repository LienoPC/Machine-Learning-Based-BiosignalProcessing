import csv
import os
import pickle

import numpy as np
import pandas as pd

from Model.DataPreprocess.SpectrogramImages import SignalPreprocess


def resample_labels(labels, fs_old, fs_new):
    new_length = int(np.ceil(len(labels)*fs_new/fs_old))

    new_labels = np.zeros(new_length, dtype=labels.dtype)

    for i in range(new_length):
        start = int(np.floor(i*fs_old/fs_new))
        end = int(np.floor((i+1)*fs_old/fs_new))

        if end > len(labels):
            end = len(labels)

        block = labels[start:end]
        if block.size:
            new_labels[i] = np.bincount(block).argmax()
        else:
            new_labels[i] = 0

    return new_labels


def window_labels(labels, window_size, stride):
    windowed = []
    for start in range(0, len(labels) - window_size + 1, stride):
        block = labels[start : start + window_size]
        counts = np.bincount(block)
        label = counts.argmax()
        windowed.append(label)

    return np.array(windowed)


def remove_invalid_labels_from_dataset():
    df = pd.read_csv("Data/WESAD/WESAD.csv", header=None, names=["img", "label"])
    df = df[df["label"].isin([1, 2])].reset_index(drop=True)
    df["label"] = np.where(df["label"] == 1, 0, 1)
    print(len(df))
    df.to_csv("Data/WESAD/WESAD_filtered.csv", index=False, header=False)

def redefine_invalid_labels_from_dataset():
    df = pd.read_csv("Data/WESAD/WESAD.csv", header=None, names=["img", "label"])

    df = df[(df["label"] != 0) & (df["label"] < 5)]
    df["label"] = np.where((df["label"] == 1) | (df["label"] > 2), 0, 1)

    print(len(df))
    df.to_csv("Data/WESAD/WESAD_redefined.csv", index=False, header=False)


def build_dataset():
    dir_list = os.listdir("Data/WESAD/")
    img_list = []
    labels_list = []

    window_length = 15
    fs = 4
    overlap = 0.5
    window_size = int(window_length * fs)
    stride = int(window_size * (1 - overlap))
    eps = 1e-6
    for subj in dir_list:
        with open(os.path.join("Data/WESAD/", subj, f"{subj}.pkl"), "rb") as file:
            data = pickle.load(file, encoding="bytes")

            gsr_wrist = np.concatenate(data[b'signal'][b'wrist'][b'EDA'])
            signal_preprocess = SignalPreprocess(4)

            labels = np.array(data[b'label'])

            labels_resampled = resample_labels(labels, 700, 4)

            assert len(gsr_wrist) == len(labels_resampled)
            labels_window = window_labels(labels_resampled, window_size, stride)
            labels_list.extend(labels_window)

            gsr_wrist = 100.0 / (gsr_wrist + eps)

            img_list.extend(signal_preprocess.entire_signal_to_scalogram_images(gsr_wrist, epoch_length=window_length, output_folder="Data/WESAD_Dataset", additional_path="Dataset/"))
            assert len(img_list) == len(labels_list)


    uniq_all, counts_all = np.unique(np.array(labels_list), return_counts=True)
    print("Accumulated labels_list distribution:")
    for u, c in zip(uniq_all, counts_all):
        print(f"  Label {u}: {c} occurrences")



    with open("Data/WESAD/WESAD.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for img, label in zip(img_list, labels_list):
            writer.writerow([img, label])


    print(f"Img list length: {len(img_list)}")
    print(f"Labels list length: {len(labels_list)}")
    # Verify if saved data is correct
    df = pd.read_csv("Data/WESAD/WESAD.csv", header=None, names=["img", "label"])

    img_list_saved = df["img"].tolist()
    label_list_saved = df["label"].astype(int).tolist()

    for img, img_saved in zip(img_list, img_list_saved):
        assert (img == img_saved)

    for label, label_saved in zip(labels_list, label_list_saved):
        assert (label == label_saved)


build_dataset()
remove_invalid_labels_from_dataset()
redefine_invalid_labels_from_dataset()