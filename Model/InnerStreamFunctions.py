import asyncio
import datetime
import re

import cv2
import pandas as pd
from pydantic import BaseModel

from scipy.signal import resample_poly

from Server.UnityStream import unity_stream
from Utility.DataLog import DataLogger
from Model.DataPreprocess.SpectrogramImages import SignalPreprocess
from fractions import Fraction
import httpx

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

def extract_signals_from_dict(sensor_data_list):
    """
    Gets a list of BiosignalData objects and return an array for each signal type
    :param sensor_data_list:
    :return:
    """
    gsr_signal = []
    ppg_signal = []
    heart_rate_signal = []
    sampling_rate = []
    timestamps = []

    for sensor_data in sensor_data_list:
        d = sensor_data
        gsr_signal.append(d.get('gsr', None))
        ppg_signal.append(d.get('ppg', None))
        heart_rate_signal.append(d.get('heart_rate', None))
        sampling_rate.append(d.get('sample_rate', None))
        timestamps.append(d.get('timestamp', None))

    return gsr_signal, ppg_signal, heart_rate_signal, sampling_rate, timestamps


async def get_sampling_freq(dataManager, window_seconds):
    """
    Extracts sampling frequency when stream starts
    :param dataManager: DataManager object to access shared queue
    :param window_seconds: dimension in seconds of the time window to process
    :return:
    """
    # Get sampling frequency when stream starts
    while True:
        first_line = dataManager.read_batch(1)
        if first_line != None and len(first_line) > 0:
            sampling_freq = first_line[0]['sample_rate']
            break
        # Flush the content of the streaming queue until now to not get old data
        dataManager.clear()
    window_samples = sampling_freq * window_seconds  # Number of samples for each window
    return sampling_freq, window_samples


async def data_processing(dataManager, websocketManager, window_seconds, overlap, model_sampling_freq, stopEvent: asyncio.Event, predict_fn=None):
    '''
    Function that manages all the streaming loop between signals and client application. Preprocesses data, applies model and stream results
    :param dataManager: object that contains the reference to the shared queue used by the input thread to read data from the biosensor
    :param websocketManager: object that contains all references to all opened websocket connections (outstream)
    :param window_seconds: dimension in seconds of the window to process
    :param overlap: percentage of overlap between windows
    :param model_sampling_freq: sampling frequency at which the predicting model works
    :param stopEvent: event called by the main thread to block the execution of the processing loop

    :return:
    '''

    try:

        if not predict_fn:
            # Get data info + flush all data received until now
            sampling_freq, window_samples = await get_sampling_freq(dataManager, window_seconds)
            if sampling_freq == 0:
                raise ValueError("Data format is not valid. Cannot find sampling frequency.")

        # Create signal preprocess object the same frequency on which the model works
        signal_preprocess = SignalPreprocess(model_sampling_freq)

        # Wait for first window to be filled
        await asyncio.sleep(window_seconds)

        # Create dataLog files
        data_logger = DataLogger("./ExperimentsLog/")
        # Fires prediction every time a window is available
        async for _ in ticker(window_seconds):
            if stopEvent.is_set():
                print("Exiting data streaming...")
                break
            await asyncio.sleep(0.01)


            if predict_fn: # TODO: Remove, test only
                await unity_stream(predict_fn(), websocketManager)
            else:
                # Reads a window_samples of data and leaves an overlap*window_samples number of elements for the next window
                read_list = dataManager.read_window_overlap(window_samples, overlap)

                if read_list and len(read_list) > 0:
                    # Call the function to elaborate the signal window
                    gsr_window, _, _, _, timestamp_window = extract_signals_from_dict(read_list)
                    # Write entire window to log file
                    for gsr, timestamp in zip(gsr_window, timestamp_window):
                        data_logger.add_raw(gsr, timestamp)

                    # Call prediction function
                    prediction = await apply_cnn_model(gsr_window, sampling_freq, signal_preprocess, data_logger)
                    # Stream prediction result to all connected clients
                    await unity_stream(prediction, websocketManager)

        data_logger.close()
    except Exception as e:
        raise e

async def ticker(interval: float):
    while True:
        yield
        await asyncio.sleep(interval)


async def apply_cnn_model(signal_window, sampling_freq, signal_preprocess, data_logger=None, scr=True):
    """
    Takes the window signal, applies preprocessing to it, and returns the prediction result
    :param signal_window:
    :param signal_preprocess:
    :param data_logger:
    :return: prediction value (0,1)
    """
    try:
        timestamp = datetime.datetime.now()
        # Resample to correct frequency
        resampled_signal = signal_preprocess.resample_epoch(signal_window, sampling_freq)
        if scr:
            # Extract SCR from signal
            resampled_signal = signal_preprocess.preprocess_signal(resampled_signal)
        #scr_signal = neurokit2.signal_filter(scr_signal, sampling_rate=signal_preprocess.fs, lowcut=0.0005, highcut=1.8)
        # Compute scalogram image
        scalogram_image = signal_preprocess.epoch_to_scalogram_image_pywt(resampled_signal)



        # Encode image
        scalogram_bgr = cv2.cvtColor(scalogram_image, cv2.COLOR_RGB2BGR)

        success, encoded_png = cv2.imencode('.png', scalogram_bgr)
        if not success:
            raise RuntimeError("Could not encode image")


        files = {'file': ('scalogram.png', encoded_png.tobytes(), 'image/png')}
        url = "http://localhost:8000/predict"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                files=files,
                timeout=5.0
            )
        if resp.status_code == 200:
            data = resp.json()
            if data_logger:
                # Logs prediction result
                data_logger.add_prediction(data['label'], scalogram_image, timestamp)
            return data['label']
        else:
            raise Exception("Error", resp.status_code, resp.text)
    except Exception as e:
        raise e


async def apply_ml_models(signal_window, sampling_freq, data_logger=None):
    """
    Takes the window signal, applies preprocessing to it, and returns the prediction result using ML models
    :param signal_window:
    :param signal_preprocess:
    :param data_logger:
    :return:
    """
    try:
        timestamp = datetime.datetime.now()
        #signal_window = signal_preprocess.clean_epoch(signal_window, sampling_freq)
        #signal_window = np.asarray(neurokit2.signal_filter(signal_window, sampling_rate=sampling_freq, lowcut=0.2, highcut=8, method='butterworth', order=8))
        window_list = [float(x) for x in signal_window.tolist()]

        payload = {
            "window": window_list,
            "sampling_rate": int(sampling_freq)
        }
        url = "http://localhost:8000/predict_ml"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json=payload,
                timeout=15.0
            )
        if resp.status_code == 200:
            data = resp.json()
            if data_logger:
                data_logger.add_prediction_ml(data['label'], timestamp)
            return data['label']
        else:
            raise Exception("Error", resp.status_code, resp.text)
    except Exception as e:
        raise e

async def apply_spectrogram_model(signal_window, sampling_freq, signal_preprocess, data_logger=None):
    """
    Takes the window signal, applies preprocessing to it, and returns the prediction result using the spectrogram model
    :param signal_window:
    :param signal_preprocess:
    :param data_logger:
    :return:
    """
    try:
        timestamp = datetime.datetime.now()
        # Resample to correct frequency
        clean_signal = signal_preprocess.resample_epoch(signal_window, sampling_freq)
        #clean_signal = neurokit2.eda.eda_clean(signal_window, sampling_rate=4)
        spectrogram_image = signal_preprocess.epoch_to_spectrogram_image(clean_signal, lowfreq=0.5, highfreq=12)

        # Encode image
        spectrogram_image = cv2.cvtColor(spectrogram_image, cv2.COLOR_RGB2BGR)
        success, encoded_png = cv2.imencode('.png', spectrogram_image)
        if not success:
            raise RuntimeError("Could not encode image")


        files = {'file': ('scalogram.png', encoded_png.tobytes(), 'image/png')}
        url = "http://localhost:8000/predict"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                files=files,
                timeout=2.0
            )
        if resp.status_code == 200:
            data = resp.json()
            if data_logger:
                data_logger.add_prediction(data['label'], spectrogram_image, timestamp)
            return data['label']
        else:
            raise Exception("Error", resp.status_code, resp.text)
    except Exception as e:
        raise e

