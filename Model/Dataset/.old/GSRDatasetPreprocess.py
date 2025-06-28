import os
import numpy as np
import scipy.signal as signal
import cv2
import scipy.io as sio
import pandas as pd

# Define parameters
fs_biopac = 2000  # Sampling frequency (2 kHz)
lowfreq = 6  # Lower frequency bound for spectrogram
highfreq = 12  # Upper frequency bound for spectrogram
freqbin = 1  # Frequency bin size

epochlength = 30  # Epoch length in seconds
overlap = 0.5  # 50% overlap

# Paths
loadfpath = "C:/Users/Prova/Desktop/Utiversinà/TheEnd/MainReference/Dataset/subject_data_biosignal/subject data/subject1.mat"
savepath = "C:/Users/Prova/Desktop/Utiversinà/TheEnd/MainReference/Dataset/subject_data_biosignal/image data"


# Load the .mat file into a Python dictionary
mat_data = sio.loadmat(loadfpath)

# Iterate through each key-value pair in the dictionary
for var_name, var_value in mat_data.items():
    # Skip MATLAB internal variables (those that start with '__')
    if var_name.startswith('__'):
        continue

    print(f"Variable: {var_name}")
    print("-" * 50)

    # Check if the variable is an array and has at most 2 dimensions
    if hasattr(var_value, 'ndim') and var_value.ndim <= 2:
        try:
            df = pd.DataFrame(var_value)
            print(df)
        except Exception as e:
            print("Error converting variable to DataFrame:", e)
            print("Raw data:")
            print(var_value)
    else:
        print("Data (raw):")
        print(var_value)

    print("\n" + "=" * 50 + "\n")
'''
# Process each subject
for subject_num in range(1, 31):  # Subjects 1 to 30
    for condition in ["pre", "S1", "S2"]:
        filename = f"subject{subject_num}_{condition}.mat"
        filepath = os.path.join(loadfpath, filename)

        if not os.path.exists(filepath):
            continue

        # Load GSR data
        mat_data = sio.loadmat(filepath)

        # Print .mat file contents
        print(f"\nContents of {filename}:")
        for var_name, var_value in mat_data.items():
            if not var_name.startswith("__"):  # Skip system variables
                print(f"Variable: {var_name}")
                print(var_value)

        gsr_signal = mat_data['data']['conductance'][0][0].squeeze()

        # Epoch segmentation
        epoch_samples = epochlength * fs_biopac
        step = int(epoch_samples * (1 - overlap))
        epochnum = (len(gsr_signal) - epoch_samples) // step + 1

        logpower_list = []
        for i in range(epochnum):
            start = i * step
            end = start + epoch_samples
            epoch_data = gsr_signal[start:end]

            # Compute spectrogram
            freqs, times, Sxx = signal.spectrogram(epoch_data, fs=fs_biopac, nperseg=epoch_samples // 30, noverlap=0,
                                                   nfft=fs_biopac, scaling='density')
            logpower = 10 * np.log10(Sxx)

            # Select frequency range
            freq_idx = np.where((freqs >= lowfreq) & (freqs <= highfreq))[0]
            logpower = logpower[freq_idx, :]
            logpower_list.append(logpower)

        logpower_total = np.stack(logpower_list, axis=-1)  # Shape: (freq, time, epochs)

        # Normalize across epochs
        logpower_mean = np.mean(logpower_total, axis=-1, keepdims=True)
        logpower_norm = logpower_total - logpower_mean

        # Save spectrograms as images
        for epoch_idx in range(logpower_norm.shape[-1]):
            img = logpower_norm[:, :, epoch_idx]
            img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize to [0,1]
            img = (img * 255).astype(np.uint8)  # Convert to 8-bit image
            img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)

            save_folder = os.path.join(savepath, "Stress" if condition != "pre" else "Normal")
            os.makedirs(save_folder, exist_ok=True)
            save_filename = f"{subject_num}_{epoch_idx}.jpg"
            cv2.imwrite(os.path.join(save_folder, save_filename), img_resized)
'''