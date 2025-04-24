import asyncio
import datetime
import re
import time

import pandas as pd
from pydantic import BaseModel
import os
import numpy as np
import scipy.signal as signal
import cv2

from DataQueueManager import DataQueueManager


class BiosignalData(BaseModel):
    heart_rate: int
    gsr: float
    ppg: float
    sample_rate: float

def downsample_data(raw_signal, timestamps, output_frequency):
    """
    Resample a raw biosignal
    :param raw_signal: array of signal samples
    :param timestamps: array of timestamp sample-associated
    :param output_frequency: the desired output frequency to resample on
    :return: returns a pandas data frame that contains the resampled signal
    """
    # Pandas' data frame from array of sampled signal
    df_signal = pd.DataFrame(
        {
            'datetime': [timestamps],
            'value': [raw_signal]
        }
    )
    df_signal.datetime = pd.to_datetime(df_signal.datetime)
    df_signal.set_index('datetime', inplace=True)
    # Resample the signal to a new frequency
    df_signal.resample(f"{(float)(1/output_frequency)}s").mean()
    return df_signal

def process_signal_to_spectrogram_images(raw_signal, fs=128, epoch_length_sec=30, overlap=0.5,
                                         lowfreq=6, highfreq=12, freqbin=1,
                                         output_size=(64, 64), output_folder='output'):
    """
    Process a raw biosignal into a grayscale spectrogram images

    Parameters:
      raw_signal: 1D array of signal data
      fs: Sampling frequency (Hz)
      epoch_length_sec: Duration of each epoch in seconds
      overlap: Percentage overlap between epochs (0 to 1)
      lowfreq: Lower frequency for spectrogram
      highfreq: Upper frequency for spectrogram
      freqbin: Frequency resolution (Hz)
      output_size: Desired output image dimensions (width, height)
      output_folder: Folder to save the output images

    Returns:
      List of normalized spectrogram arrays for each window (called epoch).
    """
    # Calculate epoch parameters
    epoch_samples = int(epoch_length_sec * fs)
    hop = int(epoch_samples * (1 - overlap))
    total_samples = len(raw_signal)
    num_epochs = int(np.floor((total_samples - epoch_samples) / hop) + 1)

    spectrograms = []

    # Frequency vector that defines the sampling rate on the magnitude
    desired_freqs = np.arange(lowfreq, highfreq + freqbin, freqbin)

    # Process each window
    for epoch_idx in range(num_epochs):
        start = epoch_idx * hop
        end = start + epoch_samples
        epoch_data = raw_signal[start:end]

        # Compute spectrogram:
        # Use a window length equal to fs
        nperseg = fs
        # Let the function use a default 50% overlap for the STFT
        f, t, ps = signal.spectrogram(epoch_data, fs=fs, window='hann', nperseg=nperseg, noverlap=None)

        # Select only the frequency bins of interest
        freq_indices = np.where((f >= lowfreq) & (f <= highfreq))[0]
        ps_selected = ps[freq_indices, :]

        # Convert power spectral density to dB scale
        logpower = 10 * np.log10(ps_selected + 1e-10)

        # Flip vertically so that low frequencies appear at the bottom
        logpower_flipped = np.flipud(logpower)
        spectrograms.append(logpower_flipped)

    # Stack all spectrograms along a new third axis:
    # shape = (n_freq_bins, n_time_bins, n_epochs)
    all_specs = np.stack(spectrograms, axis=2)

    # Compute the grand mean over all epochs
    grand_mean = np.mean(all_specs, axis=2)

    # Subtract the grand mean from each epoch
    specs_mean_sub = all_specs - grand_mean[:, :, np.newaxis]

    global_min = np.min(specs_mean_sub)
    global_max = np.max(specs_mean_sub)

    # Normalize each spectrogram to the [0, 1] range
    norm_specs = []
    for i in range(specs_mean_sub.shape[2]):
        spec = specs_mean_sub[:, :, i]
        norm_spec = (spec - global_min) / (global_max - global_min)
        norm_specs.append(norm_spec)

    os.makedirs(output_folder, exist_ok=True)

    # Process each normalized spectrogram:
    # replicate to 3 channels, resize, and save as an image
    for i, norm_spec in enumerate(norm_specs):
        img = np.stack([norm_spec] * 3, axis=-1)
        # Resize to output_size using nearest neighbor interpolation
        img_resized = cv2.resize(img, output_size, interpolation=cv2.INTER_NEAREST)
        img_uint8 = (img_resized * 255).astype(np.uint8)

        filename = os.path.join(output_folder, f"epoch_{i + 1}.jpg")
        cv2.imwrite(filename, img_uint8)

    return norm_specs

def log_to_file(obj, log_file):
    log_file = open(log_file, 'a')
    timestamp = datetime.datetime.now()
    log_file.write(f"Heart Rate: {obj['heart_rate']}; {timestamp}\n")
    log_file.write(f"PPG: {obj['ppg']}; {timestamp}\n")
    log_file.write(f"GSR: {obj['gsr']}; {timestamp}\n")
    log_file.write(f"Sample Rate: {obj['sample_rate']}; {timestamp}\n")

def log_to_queue(obj, dataManager):
    dataManager.push_single(obj)

def parse_file(filename, target_filename, num_entries=None):
    """
    Reads the first num_entries entries (each consisting of 4 lines) from the file,
    processes them into BiosignalData objects, writes these extracted lines to a new target file,
    and removes them from the original file.

    Each entry is expected to have the following format:
        Heart Rate: <int>; <timestamp>
        PPG: <float>; <timestamp>
        GSR: <float>; <timestamp>
        Sample Rate: <float>; <timestamp>

    If num_entries is None, then the entire contents of the file are processed,
    moved to the target file, and the original file is emptied.

    :param filename: Name of the file to be parsed and cleared.
    :param target_filename: Name of the file where the extracted data will be stored.
    :param num_entries: Number of objects to read from the file (each object is 4 lines),
                        if None, process the whole file.
    :return: sensor_data_list: List of the read BiosignalData objects,
             timestamp_list: List of timestamps associated with each entry
    """
    sensor_data_list = []
    timestamp_list = []

    # Read all lines from the source file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Determine how many lines to process (each entry consists of 4 lines)
    if num_entries is None:
        max_lines = len(lines)
    else:
        max_lines = min(len(lines), num_entries * 4)

    extracted_lines = lines[:max_lines]

    # Process each complete block of 4 lines in the extracted portion
    for i in range(0, max_lines, 4):
        if i + 3 < max_lines:
            hr_match = re.match(r"Heart Rate: (-?\d+); (.+)", extracted_lines[i].strip())
            ppg_match = re.match(r"PPG: ([\d\.]+); (.+)", extracted_lines[i + 1].strip())
            gsr_match = re.match(r"GSR: ([\d\.]+); (.+)", extracted_lines[i + 2].strip())
            sample_rate_match = re.match(r"Sample Rate: ([\d\.]+); (.+)", extracted_lines[i + 3].strip())

            if hr_match and ppg_match and gsr_match and sample_rate_match:
                heart_rate = int(hr_match.group(1))
                ppg = float(ppg_match.group(1))
                gsr = float(gsr_match.group(1))
                sample_rate = float(sample_rate_match.group(1))

                # Use the timestamp from the Heart Rate line; timestamps on the other lines are assumed to match
                timestamp = hr_match.group(2)
                timestamp_list.append(timestamp)

                sensor_data_list.append(BiosignalData(
                    heart_rate=heart_rate,
                    ppg=ppg,
                    gsr=gsr,
                    sample_rate=sample_rate
                ))
            else:
                print(f"Skipping invalid data block at lines {i}-{i + 3}")

    # Write the extracted (processed) lines to the target file.
    # If the target file already exists, this will overwrite its content.
    with open(target_filename, 'w') as target_file:
        target_file.writelines(extracted_lines)

    # Remove the extracted lines from the original file.
    # Here we keep only the remaining lines.
    remaining_lines = lines[max_lines:]
    with open(filename, 'w') as file:
        file.writelines(remaining_lines)

    return sensor_data_list, timestamp_list

def extract_signals_from_filedata(sensor_data_list):
    gsr_signal = []
    ppg_signal = []
    heart_rate_signal = []

    for sensor_data in sensor_data_list:
        gsr_signal.append(sensor_data.gsr)
        ppg_signal.append(sensor_data.ppg)
        heart_rate_signal.append(sensor_data.heart_rate)

    return gsr_signal, ppg_signal, heart_rate_signal


async def data_processing_mock(dataManager):
    while True:
        read_list = dataManager.read_batch(1)
        if read_list != None and len(read_list) > 0:
            log_to_file(read_list[0], log_file="Log/Stream.txt")

        await asyncio.sleep(0.5)


'''
sensor_list, timestamps = parse_file(filename="./Log/Stream.txt", target_filename="./Log/BackupStream.txt")

print(f"Sensor list: {sensor_list}")

gsr_signal, ppg_signal, heart_rate_signal = extract_signals_from_filedata(sensor_list)

print(f"GSR signals{gsr_signal}")
'''
