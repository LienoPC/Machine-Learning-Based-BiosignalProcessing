import asyncio
import datetime
import random
import re

import pandas as pd
from pydantic import BaseModel
import os
import numpy as np
import scipy.signal as signal
import cv2
from scipy.signal import resample_poly
from scipy.signal import butter

from Server.UnityStream import unity_stream
from Utility.SliderWindow import SliderWindow
from Model.DataPreprocess.SpectrogramImages import SignalPreprocess
from fractions import Fraction


class BiosignalData(BaseModel):
    heart_rate: int
    gsr: float
    ppg: float
    sample_rate: float

def downsample_data_timestamps(raw_signal, timestamps, output_frequency):
    """
    Resample a raw signal with timestamps
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


def resample_signal(x, fs_in, fs_out):
    """
    Resample from fs_in to fs_out using polyphase filtering (without timestamps)
    """
    frac = Fraction(int(fs_out), int(fs_in)).limit_denominator()
    up, down = frac.numerator, frac.denominator
    y = resample_poly(x, up, down)
    return y



def log_to_file(obj, log_file):
    log_file = open(log_file, 'a')
    timestamp = datetime.datetime.now()
    log_file.write(f"Heart Rate: {obj['heart_rate']}; {timestamp}\n")
    log_file.write(f"PPG: {obj['ppg']}; {timestamp}\n")
    log_file.write(f"GSR: {obj['gsr']}; {timestamp}\n")
    log_file.write(f"Sample Rate: {obj['sample_rate']}; {timestamp}\n")

def log_window_to_file(window, log_file):
    for line in window:
        log_to_file(line, log_file)

    log_file = open(log_file, 'a')
    log_file.write(f"//////////////////////////////////////Window//////////////////////////////////////////") # TODO: Remove this part, only for testing purpose


def log_to_queue(obj, dataManager):
    dataManager.push_single(obj)


def parse_file_no_extract(filename, num_entries=None):
    """
    Reads the first num_entries entries (each consisting of 4 lines) from the file,
    processes them into BiosignalData objects and returns them as a list

    Each entry is expected to have the following format:
        Heart Rate: <int>; <timestamp>
        PPG: <float>; <timestamp>
        GSR: <float>; <timestamp>
        Sample Rate: <float>; <timestamp>

    If num_entries is None, then the entire contents of the file are processed,
    moved to the target file, and the original file is emptied.

    :param filename: Name of the file to be parsed.
    :param num_entries: Number of objects to read from the file (each object is 4 lines),
                        if None, process the whole file.
    :return: sensor_data_list: List of the read BiosignalData objects,
             timestamp_list: List of timestamps associated with each entry
    """
    sensor_data_list = []
    timestamp_list = []

    # Regexes for each line type
    hr_re = re.compile(r"^Heart Rate:\s*(-?\d+);\s*(.+)$")
    ppg_re = re.compile(r"^PPG:\s*([\d.]+);\s*(.+)$")
    gsr_re = re.compile(r"^GSR:\s*([\d.]+);\s*(.+)$")
    sr_re = re.compile(r"^Sample Rate:\s*([\d.]+);\s*(.+)$")

    def is_separator(line):
        return line.startswith('/') and 'Window' in line

    # load & strip
    with open(filename, 'r') as f:
        raw = [ln.strip() for ln in f]

    # Drop blanks & separators early
    lines = [ln for ln in raw if ln and not is_separator(ln)]

    i = 0
    read = 0
    limit = float('inf') if num_entries is None else num_entries

    while i + 3 < len(lines) and read < limit:
        # Try to match a full 4-line record at lines[i:i+4]
        m1 = hr_re.match(lines[i])
        m2 = ppg_re.match(lines[i+1])
        m3 = gsr_re.match(lines[i+2])
        m4 = sr_re.match(lines[i+3])

        if m1 and m2 and m3 and m4:
            heart_rate = int(m1.group(1))
            ppg = float(m2.group(1))
            gsr = float(m3.group(1))
            sample_rate = float(m4.group(1))
            timestamp = m1.group(2)

            sensor_data_list.append(BiosignalData(
                heart_rate=heart_rate,
                ppg=ppg,
                gsr=gsr,
                sample_rate=sample_rate
            ))
            timestamp_list.append(timestamp)

            read += 1
            i += 4
        else:
            # No match, shift window by one
            i += 1

    print(f"Sensor data list length: {len(sensor_data_list)}")
    return sensor_data_list, timestamp_list


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
        d = sensor_data.dict()
        gsr_signal.append(d['gsr'])
        ppg_signal.append(d['ppg'])
        heart_rate_signal.append(d['heart_rate'])

    return gsr_signal, ppg_signal, heart_rate_signal



async def data_processing(dataManager, websocketManager, stopEvent: asyncio.Event):
    '''
    Read the first line and get the sampling frequency. Then define the dimension of the window
    :param dataManager: object that contains the reference to the shared queue used by the input thread to read data from the biosensor
    :param websocketManager: object that contains all references to all opened websocket connections (outstream)
    :param stopEvent: event called by the main thread to block the execution of
    :return:
    '''
    window_seconds = 15
    overlap = 0.5
    # Get sampling frequency when stream starts
    sampling_freq = 15
    while True:
        first_line = dataManager.read_batch(1)
        if first_line != None and len(first_line) > 0:
            sampling_freq = first_line[0]['sample_rate']
            break
        # Flush the content of the streaming queue until now to not get old data
        dataManager.clear()
    window_samples = sampling_freq * window_seconds # Number of samples for each window
    print(f"Read sampling freq: {sampling_freq}\n\nNumber of samples per window: {window_samples}")

    signal_preprocess = SignalPreprocess(sampling_freq)
    slider = SliderWindow()
    epoch = 0
    await asyncio.sleep(window_seconds)

    async for _ in ticker(window_seconds): # Fires prediction every time a window is available
        if stopEvent.is_set():
            print("Exiting data streaming...")
            break
        slider.tick()
        # Reads a window_samples of data and leaves an overlap*window_samples number of elements for the next window
        read_list = dataManager.read_window_overlap(window_samples, overlap)
        if read_list != None and len(read_list) > 0:
            #log_window_to_file(read_list, log_file="Log/Stream.txt")
            # Call the function to elaborate the signal window
            gsr_window, _, _ = extract_signals_from_filedata(read_list)
            await unity_stream(apply_model_mock(read_list, slider.shared_var.get()), websocketManager)
            prediction = apply_model(gsr_window, signal_preprocess, epoch)
            epoch += 1
            await unity_stream(prediction, websocketManager)


def apply_model(signal_window, signal_preprocess, epoch):
    spectrogram_image = signal_preprocess.epoch_to_spectrogram_image(signal_window)
    scalogram_image = signal_preprocess.epoch_to_scalogram_image_pywt(signal_window)
    # TODO: Remove image saving, for testing only

    fname = os.path.join("Log/Images/Scalogram", f"epoch_{epoch + 1}.png")
    cv2.imwrite(fname, cv2.cvtColor(scalogram_image, cv2.COLOR_RGB2BGR))
    fname = os.path.join("Log/Images/Spectrogram", f"epoch_{epoch + 1}.png")
    cv2.imwrite(fname, cv2.cvtColor(spectrogram_image, cv2.COLOR_RGB2BGR))

    classification_res = biased_bit(0.5)
    return classification_res


def apply_model_mock(signal_window, mean_value):
    # ELABORATE SIGNAL WINDOW
    classification_res = biased_bit(mean_value)
    return classification_res


async def ticker(interval: float):
    while True:
        yield
        await asyncio.sleep(interval)

def biased_bit(p: float) -> int:
    """
    Returns 1 with probability p, 0 with probability (1-p).
    Mean = p.
    """
    return 1 if random.random() < p else 0


def scalogram_test():
    window_seconds = 10
    overlap = 0.5
    # Get sampling frequency when stream starts

    sensor_data_list, _ = parse_file_no_extract("../Log/Stream.txt")
    if len(sensor_data_list):
        first_line = sensor_data_list[0]
        sampling_freq = first_line.sample_rate
        gsr, _, _ = extract_signals_from_filedata(sensor_data_list)
        signal_preprocess = SignalPreprocess(sampling_freq)

        signal_preprocess.entire_signal_to_scalogram_images(gsr, epoch_length=window_seconds, overlap=overlap)


scalogram_test()