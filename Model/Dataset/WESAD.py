import csv
import os
import pickle

import cv2
import neurokit2
import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import resample_poly, decimate

from Model.DataPreprocess.SpectrogramImages import SignalPreprocess


def resample_labels(labels, fs_old, fs_new, label_weights=None):

    if label_weights is None:
        max_label = labels.max()
        label_weights = {l: 1.0 for l in range(max_label + 1)}
    new_length = int(np.ceil(len(labels)*fs_new/fs_old))

    new_labels = np.zeros(new_length, dtype=labels.dtype)

    for i in range(new_length):
        start = int(np.floor(i*fs_old/fs_new))
        end = int(np.floor((i+1)*fs_old/fs_new))

        if end > len(labels):
            end = len(labels)

        block = labels[start:end]
        if block.size:
            block_weights = np.array([label_weights.get(l, 0) for l in block])
            new_labels[i] = np.bincount(block, weights=block_weights).argmax()
        else:
            new_labels[i] = 0

    return new_labels

def resample_labels_normalized(labels, fs_old, fs_new, label_weights=None):
    max_label = labels.max()
    if label_weights is None:
        max_label = labels.max()
        label_weights = {l: 1.0 for l in range(max_label + 1)}
    new_length = int(np.ceil(len(labels)*fs_new/fs_old))

    new_labels = np.zeros(new_length, dtype=labels.dtype)

    for i in range(new_length):
        start = int(np.floor(i*fs_old/fs_new))
        end = int(np.floor((i+1)*fs_old/fs_new))

        if end > len(labels):
            end = len(labels)

        block = labels[start:end]
        if block.size:
            counts = np.bincount(block, minlength=max_label+1)
            freqs = counts / block.size
            weights_array = np.array([label_weights.get(l, 0) for l in range(len(freqs))])
            scores = freqs * weights_array
            new_labels[i] = scores.argmax()
        else:
            new_labels[i] = 0

    return new_labels


def window_labels(labels, window_size, stride, label_weights=None):

    if label_weights is None:
        max_label = labels.max()
        label_weights = {l: 1.0 for l in range(max_label + 1)}

    windowed = []
    for start in range(0, len(labels) - window_size + 1, stride):
        block = labels[start : start + window_size]
        block_weights = np.array([label_weights.get(l, 0) for l in block])
        counts = np.bincount(block, weights=block_weights)
        label = counts.argmax()
        windowed.append(label)

    return np.array(windowed)

def window_labels_normalized(labels, window_size, stride, label_weights=None):
    max_label = labels.max()
    if label_weights is None:
        max_label = labels.max()
        label_weights = {l: 1.0 for l in range(max_label + 1)}

    windowed = []
    for start in range(0, len(labels) - window_size + 1, stride):
        block = labels[start : start + window_size]
        counts = np.bincount(block, minlength=max_label + 1)
        freqs = counts/float(window_size)
        weights_array = np.array([label_weights.get(l, 0) for l in range(len(freqs))])
        scores = freqs * weights_array
        label = scores.argmax()
        windowed.append(label)

    return np.array(windowed)




def remove_invalid_labels_from_dataset():
    df = pd.read_csv("Data/WESAD.csv", header=None, names=["img", "label"])
    df = df[df["label"].isin([1, 2])].reset_index(drop=True)
    df["label"] = np.where(df["label"] == 1, 0, 1)
    print(f"Normal dataset length: {len(df)}")
    df.to_csv("Data/WESAD_filtered.csv", index=False, header=False)

def redefine_invalid_labels_from_dataset():
    df = pd.read_csv("Data/WESAD.csv", header=None, names=["img", "label"])

    df = df[df["label"] < 5].reset_index(drop=True)
    df = df[df["label"] > 0].reset_index(drop=True)
    df["label"] = np.where(df["label"] != 2, 0, 1)
    print(f"Redefined dataset length: {len(df)}")
    df.to_csv("Data/WESAD_redefined.csv", index=False, header=False)


def build_dataset():
    dir_list = os.listdir("Data/WESAD/")
    img_list = []
    labels_list = []

    label_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}

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

            labels_resampled = resample_labels(labels, 700, 4, label_weights=label_weights)

            assert len(gsr_wrist) == len(labels_resampled)
            labels_window = window_labels(labels_resampled, window_size, stride, label_weights=label_weights)
            labels_list.extend(labels_window)
            gsr_wrist = 100.0 / (gsr_wrist + eps)

            img_list.extend(signal_preprocess.entire_signal_to_scalogram_images(gsr_wrist, epoch_length=window_length, output_folder="Data/WESAD_Dataset", additional_path="Dataset/"))
            assert len(img_list) == len(labels_list)


    uniq_all, counts_all = np.unique(np.array(labels_list), return_counts=True)
    print("Accumulated labels_list distribution:")
    for u, c in zip(uniq_all, counts_all):
        print(f"  Label {u}: {c} occurrences")



    with open("Data/WESAD.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for img, label in zip(img_list, labels_list):
            writer.writerow([img, label])


    print(f"Img list length: {len(img_list)}")
    print(f"Labels list length: {len(labels_list)}")
    # Verify if saved data is correct
    df = pd.read_csv("Data/WESAD.csv", header=None, names=["img", "label"])

    img_list_saved = df["img"].tolist()
    label_list_saved = df["label"].astype(int).tolist()

    for img, img_saved in zip(img_list, img_list_saved):
        assert (img == img_saved)

    for label, label_saved in zip(labels_list, label_list_saved):
        assert (label == label_saved)


def build_dataset_normalized_weight():
    dir_list = os.listdir("Data/WESAD/")
    img_list = []
    labels_list = []

    label_weights = {0: 1.0, 1: 1.0, 2: 3.0, 3: 1.0, 4: 1.0, 5: 1, 6: 1, 7: 1}

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

            labels_resampled = resample_labels_normalized(labels, 700, 4, label_weights=label_weights)

            assert len(gsr_wrist) == len(labels_resampled)
            labels_window = window_labels_normalized(labels_resampled, window_size, stride, label_weights=label_weights)
            labels_list.extend(labels_window)
            gsr_wrist = 100.0 / (gsr_wrist + eps)

            img_list.extend(signal_preprocess.entire_signal_to_scalogram_images(gsr_wrist, epoch_length=window_length, output_folder="Data/WESAD_Dataset", additional_path="Dataset/"))
            assert len(img_list) == len(labels_list)


    uniq_all, counts_all = np.unique(np.array(labels_list), return_counts=True)
    print("Accumulated labels_list distribution:")
    for u, c in zip(uniq_all, counts_all):
        print(f"  Label {u}: {c} occurrences")



    with open("Data/WESAD.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for img, label in zip(img_list, labels_list):
            writer.writerow([img, label])


    print(f"Img list length: {len(img_list)}")
    print(f"Labels list length: {len(labels_list)}")
    # Verify if saved data is correct
    df = pd.read_csv("Data/WESAD.csv", header=None, names=["img", "label"])

    img_list_saved = df["img"].tolist()
    label_list_saved = df["label"].astype(int).tolist()

    for img, img_saved in zip(img_list, img_list_saved):
        assert (img == img_saved)

    for label, label_saved in zip(labels_list, label_list_saved):
        assert (label == label_saved)


def build_dataset_chest():
    dir_list = os.listdir("Data/WESAD/")
    img_list = []
    labels_list = []

    label_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}

    window_length = 15
    fs = 25
    overlap = 0.5
    window_size = int(window_length * fs)
    stride = int(window_size * (1 - overlap))
    eps = 1e-6
    for subj in dir_list:
        with open(os.path.join("Data/WESAD/", subj, f"{subj}.pkl"), "rb") as file:
            data = pickle.load(file, encoding="bytes")

            gsr_chest = np.concatenate(data[b'signal'][b'chest'][b'EDA'])
            signal_preprocess = SignalPreprocess(fs)

            labels = np.array(data[b'label'])
            #gsr_chest = resample_poly(gsr_chest, fs, 700)
            gsr_chest = decimate(gsr_chest, q=int(700/fs), ftype='iir', zero_phase=True)
            gsr_chest = np.asarray(neurokit2.signal_filter(gsr_chest, sampling_rate=fs, lowcut=0.1, highcut=12, method="butterworth", order=8))
            labels_resampled = resample_labels(labels, 700, fs)

            assert len(gsr_chest) == len(labels_resampled)
            labels_window = window_labels(labels_resampled, window_size, stride)
            labels_list.extend(labels_window)

            gsr_chest = 100.0 / (gsr_chest + eps)

            img_list.extend(signal_preprocess.entire_signal_to_scalogram_images(gsr_chest, epoch_length=window_length, output_folder="Data/WESAD_Dataset", additional_path="Dataset/"))
            assert len(img_list) == len(labels_list)


    uniq_all, counts_all = np.unique(np.array(labels_list), return_counts=True)
    print("Accumulated labels_list distribution:")
    for u, c in zip(uniq_all, counts_all):
        print(f"  Label {u}: {c} occurrences")



    with open("Data/WESAD.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for img, label in zip(img_list, labels_list):
            writer.writerow([img, label])


    print(f"Img list length: {len(img_list)}")
    print(f"Labels list length: {len(labels_list)}")
    # Verify if saved data is correct
    df = pd.read_csv("Data/WESAD.csv", header=None, names=["img", "label"])

    img_list_saved = df["img"].tolist()
    label_list_saved = df["label"].astype(int).tolist()

    for img, img_saved in zip(img_list, img_list_saved):
        assert (img == img_saved)

    for label, label_saved in zip(labels_list, label_list_saved):
        assert (label == label_saved)


def build_dataset_spectrogram():
    dir_list = os.listdir("Data/WESAD/")
    img_list = []
    labels_list = []

    label_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}

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

            labels_resampled = resample_labels(labels, 700, fs, label_weights=label_weights)

            assert len(gsr_wrist) == len(labels_resampled)
            labels_window = window_labels(labels_resampled, window_size, stride, label_weights=label_weights)
            labels_list.extend(labels_window)
            gsr_wrist = 100.0 / (gsr_wrist + eps)

            img_list.extend(signal_preprocess.entire_signal_to_spectrogram_images(gsr_wrist, epoch_length_sec=window_length, output_folder="Data/WESAD_Dataset_Spectrogram", additional_path="Dataset/"))
            assert len(img_list) == len(labels_list)


    uniq_all, counts_all = np.unique(np.array(labels_list), return_counts=True)
    print("Accumulated labels_list distribution:")
    for u, c in zip(uniq_all, counts_all):
        print(f"  Label {u}: {c} occurrences")



    with open("Data/WESAD_spectrogram.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for img, label in zip(img_list, labels_list):
            writer.writerow([img, label])


    print(f"Img list length: {len(img_list)}")
    print(f"Labels list length: {len(labels_list)}")
    # Verify if saved data is correct
    df = pd.read_csv("Data/WESAD_spectrogram.csv", header=None, names=["img", "label"])

    img_list_saved = df["img"].tolist()
    label_list_saved = df["label"].astype(int).tolist()

    for img, img_saved in zip(img_list, img_list_saved):
        assert (img == img_saved)

    for label, label_saved in zip(labels_list, label_list_saved):
        assert (label == label_saved)

def build_dataset_spectrogram_chest():
    csvf = "Data/WESAD_spectrogram.csv"
    dir_list = os.listdir("Data/WESAD/")
    img_list = []
    labels_list = []

    label_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}

    window_length = 15
    fs = 25
    overlap = 0.5
    window_size = int(window_length * fs)
    stride = int(window_size * (1 - overlap))
    eps = 1e-6
    for subj in dir_list:
        with open(os.path.join("Data/WESAD/", subj, f"{subj}.pkl"), "rb") as file:
            data = pickle.load(file, encoding="bytes")

            gsr_chest = np.concatenate(data[b'signal'][b'chest'][b'EDA'])
            signal_preprocess = SignalPreprocess(fs)

            labels = np.array(data[b'label'])
            # gsr_chest = resample_poly(gsr_chest, 128, 700)
            # gsr_chest = signal_preprocess.clean_epoch(gsr_chest, 700)
            gsr_chest = decimate(gsr_chest, q=int(700 / fs), ftype='iir', zero_phase=True)
            gsr_chest = np.asarray(
                neurokit2.signal_filter(gsr_chest, sampling_rate=fs, lowcut=0.2, highcut=10, method="butterworth",
                                        order=8))
            labels_resampled = resample_labels_normalized(labels, 700, fs, label_weights=label_weights)

            assert len(gsr_chest) == len(labels_resampled)
            labels_window = window_labels_normalized(labels_resampled, window_size, stride, label_weights=label_weights)
            labels_list.extend(labels_window)

            gsr_chest = 100.0 / (gsr_chest + eps)

            img_list.extend(signal_preprocess.entire_signal_to_spectrogram_images(gsr_chest, epoch_length_sec=window_length, output_folder="Data/WESAD_Dataset_Spectrogram", additional_path="Dataset/"))
            assert len(img_list) == len(labels_list)


    uniq_all, counts_all = np.unique(np.array(labels_list), return_counts=True)
    print("Accumulated labels_list distribution:")
    for u, c in zip(uniq_all, counts_all):
        print(f"  Label {u}: {c} occurrences")



    with open("Data/WESAD_spectrogram.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for img, label in zip(img_list, labels_list):
            writer.writerow([img, label])


    print(f"Img list length: {len(img_list)}")
    print(f"Labels list length: {len(labels_list)}")
    # Verify if saved data is correct
    df = pd.read_csv("Data/WESAD_spectrogram.csv", header=None, names=["img", "label"])

    img_list_saved = df["img"].tolist()
    label_list_saved = df["label"].astype(int).tolist()

    for img, img_saved in zip(img_list, img_list_saved):
        assert (img == img_saved)

    for label, label_saved in zip(labels_list, label_list_saved):
        assert (label == label_saved)



def build_SCR_dataset():
    dir_list = os.listdir("Data/WESAD/")
    img_list = []
    labels_list = []

    label_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}

    window_length = 15
    fs = 4
    overlap = 0.2
    window_size = int(window_length * fs)
    stride = int(window_size * (1 - overlap))
    eps = 1e-6
    for subj in dir_list:
        with open(os.path.join("Data/WESAD/", subj, f"{subj}.pkl"), "rb") as file:
            data = pickle.load(file, encoding="bytes")

            gsr_wrist = np.concatenate(data[b'signal'][b'wrist'][b'EDA'])
            signal_preprocess = SignalPreprocess(4)

            labels = np.array(data[b'label'])

            labels_resampled = resample_labels(labels, 700, 4, label_weights=label_weights)

            assert len(gsr_wrist) == len(labels_resampled)
            labels_window = window_labels(labels_resampled, window_size, stride, label_weights=label_weights)
            labels_list.extend(labels_window)
            gsr_wrist = 100.0 / (gsr_wrist + eps)

            scr_signal = neurokit2.eda_phasic(gsr_wrist, fs)
            scr_signal = np.asarray(scr_signal["EDA_Phasic"])

            img_list.extend(signal_preprocess.entire_signal_to_scalogram_images(scr_signal, epoch_length=window_length, output_folder="Data/WESAD_Dataset", additional_path="Dataset/", overlap=overlap))
            assert len(img_list) == len(labels_list)


    uniq_all, counts_all = np.unique(np.array(labels_list), return_counts=True)
    print("Accumulated labels_list distribution:")
    for u, c in zip(uniq_all, counts_all):
        print(f"  Label {u}: {c} occurrences")



    with open("Data/WESAD.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for img, label in zip(img_list, labels_list):
            writer.writerow([img, label])


    print(f"Img list length: {len(img_list)}")
    print(f"Labels list length: {len(labels_list)}")
    # Verify if saved data is correct
    df = pd.read_csv("Data/WESAD.csv", header=None, names=["img", "label"])

    img_list_saved = df["img"].tolist()
    label_list_saved = df["label"].astype(int).tolist()

    for img, img_saved in zip(img_list, img_list_saved):
        assert (img == img_saved)

    for label, label_saved in zip(labels_list, label_list_saved):
        assert (label == label_saved)

def build_SCR_chest_dataset():
    dir_list = os.listdir("Data/WESAD/")
    img_list = []
    labels_list = []

    label_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}

    window_length = 15
    fs = 25
    overlap = 0.2
    window_size = int(window_length * fs)
    stride = int(window_size * (1 - overlap))
    eps = 1e-6
    for subj in dir_list:
        with open(os.path.join("Data/WESAD/", subj, f"{subj}.pkl"), "rb") as file:
            data = pickle.load(file, encoding="bytes")

            gsr_chest = np.concatenate(data[b'signal'][b'chest'][b'EDA'])
            signal_preprocess = SignalPreprocess(fs)
            gsr_chest = decimate(gsr_chest, q=int(700 / fs), ftype='iir', zero_phase=True)
            gsr_chest = np.asarray(
                neurokit2.signal_filter(gsr_chest, sampling_rate=fs, lowcut=0.0005, highcut=10, method="butterworth",
                                        order=2))

            labels = np.array(data[b'label'])

            labels_resampled = resample_labels(labels, 700, fs, label_weights=label_weights)



            assert len(gsr_chest) == len(labels_resampled)
            labels_window = window_labels(labels_resampled, window_size, stride, label_weights=label_weights)
            labels_list.extend(labels_window)
            gsr_chest = 100.0 / (gsr_chest + eps)

            scr_signal = neurokit2.eda_phasic(gsr_chest, fs)
            scr_signal = np.asarray(scr_signal["EDA_Phasic"])

            img_list.extend(signal_preprocess.entire_signal_to_scalogram_images(scr_signal, epoch_length=window_length, output_folder="Data/WESAD_Dataset", additional_path="Dataset/", overlap=overlap))
            assert len(img_list) == len(labels_list)


    uniq_all, counts_all = np.unique(np.array(labels_list), return_counts=True)
    print("Accumulated labels_list distribution:")
    for u, c in zip(uniq_all, counts_all):
        print(f"  Label {u}: {c} occurrences")



    with open("Data/WESAD.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for img, label in zip(img_list, labels_list):
            writer.writerow([img, label])


    print(f"Img list length: {len(img_list)}")
    print(f"Labels list length: {len(labels_list)}")
    # Verify if saved data is correct
    df = pd.read_csv("Data/WESAD.csv", header=None, names=["img", "label"])

    img_list_saved = df["img"].tolist()
    label_list_saved = df["label"].astype(int).tolist()

    for img, img_saved in zip(img_list, img_list_saved):
        assert (img == img_saved)

    for label, label_saved in zip(labels_list, label_list_saved):
        assert (label == label_saved)


def windowing_padding():
    dir_list = os.listdir("Data/WESAD/")
    window_length = 15
    fs = 4
    overlap = 0.2
    window_size = int(window_length * fs)
    eps = 1e-6
    label_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}

    img_list = []
    for subj in dir_list:
        with open(os.path.join("Data/WESAD/", subj, f"{subj}.pkl"), "rb") as file:
            data = pickle.load(file, encoding="bytes")

            gsr_wrist = np.concatenate(data[b'signal'][b'wrist'][b'EDA'])
            signal_preprocess = SignalPreprocess(4)

            gsr_wrist = 100.0 / (gsr_wrist + eps)

            scr_signal = neurokit2.eda_phasic(gsr_wrist, fs)
            scr_signal = np.asarray(scr_signal["EDA_Phasic"])

            img_list.extend(signal_preprocess.entire_signal_to_scalogram_images(scr_signal, epoch_length=window_length,
                                                                                output_folder="Data/Test",
                                                                                additional_path="Dataset/",
                                                                                overlap=overlap))

            hop = int(window_size * (1 - overlap))
            total = len(gsr_wrist)
            n_epochs = int(np.floor((total - window_size) / hop) + 1)
            for idx in range(n_epochs):
                start = idx * hop
                epoch = gsr_wrist[start:start + window_size]
                resampled = signal_preprocess.resample_epoch(epoch, fs)
                scr = signal_preprocess.preprocess_signal(resampled)

                # Get scalogram image
                image = signal_preprocess.epoch_to_scalogram_image_pywt(scr)
                # Save on file
                fname = f"Data/Windowed/epoch_{idx}.png"
                cv2.imwrite(fname, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

#build_SCR_chest_dataset()
#remove_invalid_labels_from_dataset()
#redefine_invalid_labels_from_dataset()

#windowing_padding()