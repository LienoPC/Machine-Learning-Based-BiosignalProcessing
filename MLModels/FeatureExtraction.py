import warnings

import pywt
import sklearn.preprocessing
import neurokit2
import scipy
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler


def extract_features(epoch_signal, sampling_rate):

    #eda_signals, info = neurokit2.eda_process(epoch_signal, sampling_rate=sampling_rate)
    #neurokit2.eda_plot(eda_signals, info)
    #plt.show()
    #epoch_signal = neurokit2.signal_filter(epoch_signal, sampling_rate=sampling_rate, lowcut=0.2, highcut=1.9, method='butterworth', order=1)

    #scr_signal = neurokit2.eda_phasic(epoch_signal, sampling_rate)
    #scr_signal = np.asarray(scr_signal["EDA_Phasic"])

    scr_signal = epoch_signal

    if len(scr_signal) == 0:
        return (0, 0, 0, 0, 0, 0)

    mean_scr = np.mean(scr_signal, axis=0)
    max_scr = np.max(scr_signal, axis=0)
    min_scr = np.min(scr_signal, axis=0)
    range_scr = max_scr - min_scr
    skeweness_scr = scipy.stats.skew(scr_signal)
    kurtosis_scr = scipy.stats.kurtosis(scr_signal)

    return (mean_scr, max_scr, min_scr, range_scr, skeweness_scr, kurtosis_scr)

def extract_window_features_list(signal, sampling_rate, window_length, stride, n_epochs):
    features_array = []
    for idx in range(n_epochs):
        start = idx * stride
        epoch = signal[start:start + window_length]
        features = extract_features(epoch, sampling_rate=sampling_rate)
        features_array.append(features)

    return features_array

def normalize_signal(signal):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(signal.reshape(-1, 1)).flatten()

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)


def swt_denoise(signal, wavelet='db4', level=3):
    coeffs = pywt.swt(signal, wavelet, level=level)

    denoised_coeffs = [(cA, np.zeros_like(cD)) for cA, cD in coeffs]
    return pywt.iswt(denoised_coeffs, wavelet)



def swt_denoise_safe(signal, wavelet='db4', level=3, method='soft', pad_mode='reflect'):
    """
    Safe SWT denoising:
      - clamps requested level to pywt.dwt_max_level if needed
      - performs SWT and soft/hard-thresholds detail coeffs
      - returns denoised signal with the same length as `signal`
      - returns used_level via warnings if different from requested

    Parameters
    ----------
    signal : array-like, 1D
    wavelet : str or pywt.Wavelet
    level : int, requested SWT decomposition level
    method : 'soft' or 'hard' thresholding
    pad_mode : if padding is necessary, mode passed to np.pad (default 'reflect')

    Returns
    -------
    denoised : np.ndarray (same length as signal)
    """
    sig = np.asarray(signal).ravel()
    if sig.size == 0:
        raise ValueError("Empty signal passed to swt_denoise_safe")

    w = pywt.Wavelet(wavelet)

    # Compute allowed max level
    try:
        max_allowed = pywt.dwt_max_level(len(sig), w.dec_len)
    except Exception:
        # conservative fallback
        max_allowed = int(np.floor(np.log2(len(sig)))) if len(sig) > 0 else 0

    used_level = min(level, max_allowed)
    if used_level < 1:
        # Nothing to do
        warnings.warn(f"swt_denoise_safe: signal too short for SWT (len={len(sig)}). Returning original signal.")
        return sig

    if used_level != level:
        warnings.warn(f"swt_denoise_safe: requested level {level} > max_allowed {max_allowed}. Using level={used_level}.")

    # If pywt.swt still throws for some reason, catch and try fallback
    try:
        coeffs = pywt.swt(sig, wavelet, level=used_level)
    except ValueError as e:
        # As a fallback try padding minimally until pywt.swt accepts the request (rare)
        # Pad by reflection up to a safe limit
        max_pad = 4 * len(sig)   # don't pad forever
        padded = sig.copy()
        pad_iters = 0
        while True:
            pad_iters += 1
            padded = np.pad(padded, (1, 1), mode=pad_mode)
            try:
                coeffs = pywt.swt(padded, wavelet, level=used_level)
                break
            except ValueError:
                if len(padded) > max_pad:
                    # give up: reduce used_level and retry
                    used_level = max(1, used_level - 1)
                    warnings.warn("swt_denoise_safe: heavy padding failed — reducing used_level to " + str(used_level))
                    coeffs = pywt.swt(sig, wavelet, level=used_level)
                    break

    # soft/hard thresholding of detail coefficients
    denoised_coeffs = []
    for (cA, cD) in coeffs:
        # estimate noise sigma from median absolute deviation of detail
        sigma = np.median(np.abs(cD)) / 0.6745 if cD.size > 0 else 0.0
        # universal threshold
        thr = sigma * np.sqrt(2 * np.log(max(1, cD.size)))
        if method == 'soft':
            cD_t = np.sign(cD) * np.maximum(np.abs(cD) - thr, 0.0)
        elif method == 'hard':
            cD_t = cD * (np.abs(cD) > thr)
        else:
            raise ValueError("method must be 'soft' or 'hard'")
        denoised_coeffs.append((cA, cD_t))

    denoised = pywt.iswt(denoised_coeffs, wavelet)

    # If we padded inside the try/except above, we need to return the central piece matching original length
    if len(denoised) != len(sig):
        start = (len(denoised) - len(sig)) // 2
        denoised = denoised[start:start + len(sig)]

    return denoised

def separate_scl_scr(signal):
    signal = signal.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=5)
    gmm.fit(signal)
    labels = gmm.predict(signal)

    if gmm.covariances_[0][0][0] > gmm.covariances_[1][0][0]:
        scr_label = 0
    else:
        scr_label = 1
    scr_signal = signal[labels == scr_label].flatten()
    return scr_signal