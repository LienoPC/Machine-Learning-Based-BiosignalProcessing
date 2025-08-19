import os
import random

import cv2
import matplotlib.pyplot as plt
import neurokit2
import numpy as np

from Model.InnerStreamFunctions import apply_cnn_model, parse_file_no_extract, extract_signals_from_dict
from Utility.AvroReader import read_and_plot

from Model.DataPreprocess.SpectrogramImages import SignalPreprocess, plot_signal, plot_signal_nosave

def apply_model_mock(mean_value):
    # ELABORATE SIGNAL WINDOW
    classification_res = biased_bit(mean_value)
    return classification_res



def biased_bit(p: float) -> int:
    """
    Returns 1 with probability p, 0 with probability (1-p).
    Mean = p.
    """
    return 1 if random.random() < p else 0


def scalogram_test():
    epoch_length = 15
    overlap = 0.5
    # Get sampling frequency when stream starts
    sensor_data_list, _ = parse_file_no_extract("./Log/Stream/Stream.txt")
    first_line = sensor_data_list[0]
    sampling_freq = first_line.sample_rate
    if len(sensor_data_list):
        gsr, _, _, _, _ = extract_signals_from_dict(sensor_data_list)
        total = len(gsr)
        signal_preprocess = SignalPreprocess(32)
        epoch_samples = int(epoch_length * sampling_freq)
        hop = int(epoch_samples * (1 - overlap))
        n_epochs = int(np.floor((total - epoch_samples) / hop) + 1)
        os.makedirs("./Log/Stream/Images", exist_ok=True)

        for idx in range(n_epochs):
            start = idx * hop
            epoch = gsr[start:start + epoch_samples]
            clean_signal = signal_preprocess.resample_epoch(epoch, sampling_freq)
            scalogram_image = signal_preprocess.epoch_to_scalogram_image_pywt(clean_signal, num_scales=32, freq_min=0.5, freq_max=8)
            fname = "./Log/Stream/Images/" + f"epoch_{idx}.png"
            cv2.imwrite(fname, cv2.cvtColor(scalogram_image, cv2.COLOR_RGB2BGR))
            print(f"Saved at {fname}")


async def dataset_forward_pass_test():

    sensor_data_list, _ = parse_file_no_extract("./Model/Log/Stream/Stream.txt")
    epoch_length = 15
    overlap = 0.2
    first_line = sensor_data_list[0]
    sampling_freq = first_line.sample_rate

    epoch_samples = int(epoch_length * sampling_freq)
    hop = int(epoch_samples * (1 - overlap))
    eps = 1e-6
    predictions = []
    if len(sensor_data_list):
        gsr, _, _, _, _ = extract_signals_from_dict(sensor_data_list)
        gsr = np.asarray(gsr)

        eda_signals, info = neurokit2.eda_process(gsr, sampling_rate=int(sampling_freq))
        neurokit2.eda_plot(eda_signals, info)
        plt.show()
        total = len(gsr)
        signal_preprocess = SignalPreprocess(4)
        n_epochs = int(np.floor((total - epoch_samples) / hop) + 1)

        for idx in range(n_epochs):
            start = idx * hop
            epoch = np.asarray(gsr[start:start + epoch_samples])
            prediction = await apply_cnn_model(epoch, sampling_freq, signal_preprocess, scr=False)
            predictions.append(prediction)
            idx += 1

    plot_signal_nosave(predictions, title="Predicted value",
                        xlabel="Time Samples", ylabel="Prediction")
    plot_signal(predictions, os.path.join("Model/Log/Stream/", "Predictions"), title="Predicted value", xlabel="Time Samples", ylabel="Prediction")


async def embrace_forward_pass_plot():
    dir_path = "Utility/EmbraceData/AlessandroVisconti/NoStressScenario/"
    sampling_freq = 4

    gsr_array = read_and_plot(["Utility/csv/1-1-0_1753974072.avro.csv"],
                        "2025-07-31 17:17:00.031178", "2025-07-31 17:27:00.715103", dir_path, sampling_freq, False)

    gsr_array = np.asarray(neurokit2.signal_filter(gsr_array, sampling_rate=sampling_freq, lowcut=0.0005, highcut=1.9, method='butterworth', order=2))

    eda_signals, info = neurokit2.eda_process(gsr_array, sampling_rate=sampling_freq)
    neurokit2.eda_plot(eda_signals, info)
    neuro_analysis = plt.gcf()
    neuro_analysis.savefig(f"{dir_path}Analysis.png")
    eps = 1e-6
    gsr_array = 100/(gsr_array + eps)
    #gsr_array = resample_poly(gsr_array, up=sampling_freq, down=4)
    epoch_length = 15
    overlap = 0.2
    epoch_samples = int(epoch_length * sampling_freq)
    hop = int(epoch_samples * (1 - overlap))

    predictions = []
    if len(gsr_array):

        total = len(gsr_array)
        signal_preprocess = SignalPreprocess(sampling_freq)
        n_epochs = int(np.floor((total - epoch_samples) / hop) + 1)

        for idx in range(n_epochs):
            start = idx * hop
            epoch = gsr_array[start:start + epoch_samples]
            prediction = await apply_cnn_model(epoch, sampling_freq, signal_preprocess)
            predictions.append(prediction)
            idx += 1

    #plot_signal_nosave(predictions, title="Predicted value", xlabel="Time Samples", ylabel="Prediction")
    plot_signal(predictions, os.path.join(dir_path, "Predictions"), title="Predicted value", xlabel="Time Samples", ylabel="Prediction")


async def random_prediction_test():
    sampling_freq = 128
    gsr_array = np.asarray(neurokit2.eda_simulate(180, sampling_rate=sampling_freq, scr_number=1, drift=-0.0001, noise=0.05))
    eda_signals, info = neurokit2.eda_process(gsr_array, sampling_rate=sampling_freq)
    neurokit2.eda_plot(eda_signals, info)
    plt.show()
    eps = 1e-6
    gsr_array = 100/(gsr_array + eps)
    #gsr_array = neurokit2.eda_clean(gsr_array, sampling_rate=sampling_freq)
    #gsr_array = resample_poly(gsr_array, up=sampling_freq, down=4)
    epoch_length = 15
    overlap = 0.2
    epoch_samples = int(epoch_length * sampling_freq)
    hop = int(epoch_samples * (1 - overlap))

    predictions = []
    if len(gsr_array):

        total = len(gsr_array)
        signal_preprocess = SignalPreprocess(4)
        n_epochs = int(np.floor((total - epoch_samples) / hop) + 1)

        for idx in range(n_epochs):
            start = idx * hop
            epoch = gsr_array[start:start + epoch_samples]
            prediction = await apply_cnn_model(epoch, sampling_freq, signal_preprocess, scr=True)
            predictions.append(prediction)
            idx += 1

    plot_signal_nosave(predictions, title="Predicted value",
                       xlabel="Time Samples", ylabel="Prediction")



def log_window_to_file(window, log_file):
    for line in window:
        log_to_file(line, log_file)

    log_file = open(log_file, 'a')
    log_file.write(f"//////////////////////////////////////Window//////////////////////////////////////////") # TODO: Remove this part, only for testing purpose
