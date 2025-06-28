import pickle

import numpy as np

from Model.DataPreprocess.SpectrogramImages import SignalPreprocess
from Model.InnerStreamFunctions import resample_signal


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


with open("Data/WESAD/S2/S2.pkl", "rb") as file:
    data = pickle.load(file, encoding="bytes")

    gsr_wrist = np.concatenate(data[b'signal'][b'wrist'][b'EDA'])
    signal_preprocess = SignalPreprocess(4)

    gsr_chest = np.concatenate(data[b'signal'][b'chest'][b'EDA'])
    eda_resampled_chest = resample_signal(data[b'signal'][b'wrist'][b'EDA'], 700, 4)

    labels = np.array(data[b'label'])
    labels_resampled = resample_labels(labels, 700, 4)
    print(labels.min(), labels.max())
    labels_list = np.where(labels_resampled > 0.5, 1, 0)
    print(labels_resampled.min(), labels_resampled.max())
    gsr_wrist = 1/gsr_wrist * 100
    print(labels_resampled)
    #img_list = signal_preprocess.entire_signal_to_scalogram_images(gsr_wrist)


