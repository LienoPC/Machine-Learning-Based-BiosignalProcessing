import os
import numpy as np
import scipy.signal as signal
import cv2
import pywt
from pandas.core.dtypes.inference import is_integer
from scipy.signal import butter, filtfilt, decimate
from scipy.signal import lfilter_zi
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import neurokit2 as nk

class SignalPreprocess():

    def __init__(self, fs, lowcut = 0.5, highcut=6, order=6):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs


    def initialize_filter(self, initial_value):
        '''
        Initialize the bandpass filter, to be called the first time before starting filtering. Sets the zi filter's state to be used in each call
        :param initial_value: First value of the time-series signal
        '''
        self.b, self.a = self.butter_bandpass_filter_definition(6)
        self.zi = lfilter_zi(self.b, self.a) * initial_value

    def butter_bandpass_filter_definition(self, order=6):
        '''
        Creates a butterworth bandpass filter. Returns polynomials (numerator and denominator) of the filter.
        :param fs: Sampling frequency (in Hz)
        :param lowcut: Lower frequency
        :param highcut: Higher frequency
        :param order: Filter order
        '''
        eps = 1e-8
        nyq_freq = 0.5 * self.fs
        # Normalize cutoffs of the Nyquist frequency
        low_freq = self.lowcut / nyq_freq
        if(self.highcut > nyq_freq):
            self.highcut = nyq_freq - eps

        high_freq = self.highcut / nyq_freq
        print(f"Normalized low_freq: {low_freq} and high_freq: {high_freq}")
        # Compute filter coefficients
        b, a = butter(order, [low_freq, high_freq], btype='band', output='ba')
        return b, a


    def apply_bandpass_filter(self, signal_samples):
        '''
        :param signal_samples: 1D Time signal to be filtered
        :return: The filtered 1D time signal result from the bandpass filter
        '''
        if not hasattr(self, 'zi'):
            self.initialize_filter(signal_samples[0])
        filtered_signal = filtfilt(self.b, self.a, signal_samples)
        return filtered_signal

    def epoch_to_spectrogram_image(self, epoch_data, lowfreq=None, highfreq=None, freqbin=1, output_size=(224, 224)):
        '''
        Turn one window of raw signal (epoch_data) into a 3-channel image spectrogram
        :param epoch_data: 1D array of raw signal data
        :param fs: Sampling frequency (Hz)
        :param lowfreq: Lower frequency for spectrogram
        :param highfreq: Higher frequency for spectrogram
        :param freqbin: Frequency resolution (Hz)
        :param output_size: Desired output image dimensions (width, height)
        :return: Spectrogram image computed
        '''

        lowfreq = lowfreq or self.lowcut
        highfreq = highfreq or self.highcut
        # Filter the 1D time signal
        epoch_data = self.apply_bandpass_filter(epoch_data)

        # STFT-based spectrogram
        f, t, ps = signal.spectrogram(epoch_data, self.fs, window='hann', nperseg=self.fs, noverlap=None)

        # Mantain only interesting frequencies
        mask = (f >= lowfreq) & (f <= highfreq)
        ps = ps[mask, :]

        # db-scale
        logp = 10 * np.log10(ps + 1e-10)

        # Flip low frequencies at the bottom
        logp = np.flip(logp)

        # Normalize to [0,1]
        logp = (logp - logp.min()) / (logp.max() - logp.min())

        # Create the image
        img = np.stack([logp] * 3, axis=-1)
        img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
        img = (img * 255).astype(np.uint8)

        return img

    def entire_signal_to_spectrogram_images(self, raw_signal, epoch_length_sec=15, overlap=0.5,
                                             lowfreq=0.5, highfreq=6, freqbin=1,
                                             output_size=(224, 224), output_folder='Log/output_spectrograms'):
        """
        Process an entire vector of raw biosignal into a grayscale spectrogram images. Saves results in a folder to be used later

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
        epoch_samples = int(epoch_length_sec * self.fs)
        hop = int(epoch_samples * (1 - overlap))
        total_samples = len(raw_signal)
        num_epochs = int(np.floor((total_samples - epoch_samples) / hop) + 1)

        spectrograms = []

        # Process each window
        for epoch_idx in range(num_epochs):
            start = epoch_idx * hop
            end = start + epoch_samples
            epoch_data = raw_signal[start:end]

            # Compute spectrogram:
            # Use a window length equal to fs
            nperseg = self.fs
            # Let the function use a default 50% overlap for the STFT
            f, t, ps = signal.spectrogram(epoch_data, fs=self.fs, window='hann', nperseg=nperseg, noverlap=None)

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


    def compute_morlet_cwt(self, raw_signal, fs, num_scales=12, freq_min=0.5, freq_max=6, w=6.0):
        '''
        Compute the continuous wavelet transform (using morlet wavelets)
        :param raw_signal: 1D Array of input signal
        :param fs: Sampling frequency in Hz
        :param num_scales: Number of wavelet scales
        :param freq_min: Minimum center frequency for the lowest scale in Hz
        :param freq_max: Maximum center frequency for the highest scale in Hz
        :param w: Morlet parameter
        :return: 2D complex array with shape (scales, len(raw_signal)) of wavelet coefficients.
        1D Array of scales used for each row of the coefficients and 1D Array of center frequencies of each scale
        '''

        # Center frequencies evenly spaced between freq_min and freq_max
        center_freqs = np.linspace(freq_min, freq_max, num_scales)

        # Convert center frequencies to scales
        scales = (w * fs) / (2 * np.pi * center_freqs)
        # Compute CWT with morlet
        coefficients = signal.cwt(raw_signal, signal.morlet2, widths=scales, w=w)

        return coefficients, scales, center_freqs

    def cwt_to_scalogram_image(self, coefficients, times=None, vmin=None, vmax=None, cmap_name='jet', n_colors=128, output_size=(224,224), interpolation=cv2.INTER_LINEAR, epoch_data=None):
        '''
        Convert CWT coefficients into an RGB scalogram image using discrete jet color map (128 colors)

        :param coefficients: Complex CWT coefficients, output of signal.cwt; Shape: (n_scales, len(raw_signal))
        :param freqs: Center frequencies corresponding to each scale
        :param times: 1D array of time. If none, columns are considered in order
        :param vmin: Min magnitude value to normalize colormap
        :param vmax: Max magnitude value to normalize colormap
        :param cmap_name: Name of the Matplotlib colormap to use ('jet' default)
        :param n_colors: Number of discrete colors in the map (128 default)
        :param output_size: Size of the created image
        :param interpolation: Type of interpolation used for resize
        :return: The scalogram as an RGB uint8 image
        '''

        # Image is created on magnitude
        magnitude = np.abs(coefficients)

        # Normalize to [0,1] value
        if vmin is None:
            vmin = magnitude.min()
        if vmax is None:
            vmax = magnitude.max()
        normalized = np.clip((magnitude - vmin) / (vmax - vmin), 0, 1)

        #self.plot_scalogram(normalized, freqs, np.linspace(0,(len(epoch_data)-1)/self.fs,len(epoch_data), endpoint=True))
        # Compute colormap
        cmap = plt.get_cmap(cmap_name, n_colors)
        rgba_img = cmap(normalized)
        rgb_img = (rgba_img[..., :3] * 255).astype(np.uint8)
        rgb_img = np.flipud(rgb_img)

        # Resize
        if output_size is not None:
            rgb_img = cv2.resize(rgb_img, output_size, interpolation=interpolation)

        return rgb_img

    def epoch_to_scalogram_image_pywt(self, epoch_data, num_scales=32, freq_min=0.5, freq_max=4):
        #epoch_data = self.apply_bandpass_filter(epoch_data)
        wavelet = pywt.ContinuousWavelet('cmor1.0-1.5')
        freqs = np.linspace(freq_min, freq_max, num_scales)

        fc = pywt.central_frequency(wavelet)
        scales = fc * self.fs / freqs
        pad_len = len(epoch_data)
        data_p = np.pad(epoch_data, pad_width=pad_len, mode='symmetric')
        coef_p, freq = pywt.cwt(data_p, scales, sampling_period=1/self.fs, wavelet=wavelet, method='fft')
        coef = coef_p[:, pad_len: pad_len + pad_len]
        scalogram_image = self.cwt_to_scalogram_image(coef, freq, epoch_data=epoch_data)
        return scalogram_image


    def epoch_to_scalogram_image(self, epoch_data):
        # Apply pass filter
        epoch_data = self.apply_bandpass_filter(epoch_data)
        # Compute cwt
        coeff, scales, center_freqs = self.compute_morlet_cwt(epoch_data, self.fs)

        # Compute image
        scalogram_image = self.cwt_to_scalogram_image(coeff, center_freqs, epoch_data=epoch_data)

        return scalogram_image

    def clean_epoch(self, epoch_data, fs_data):
        if fs_data == 4:
            return epoch_data
        final_signal = np.asarray(nk.eda_clean(epoch_data, sampling_rate=fs_data, method='neurokit'))
        # Resample data to model frequency
        if fs_data > self.fs and float.is_integer(fs_data/self.fs):
            factor = int(fs_data / self.fs)
            stages = []
            if factor == 32:
                stages = [4, 8]
            elif factor == 24:
                stages = [4, 6]
            elif factor == 16:
                stages = [4, 4]
            if len(stages) > 0:
                res_epoch = epoch_data
                for q in stages:
                    res_epoch = decimate(res_epoch, q=q, ftype='iir', zero_phase=True)
                return res_epoch
            else:
                raise ValueError(f"Cannot resample from fs_data={fs_data} Hz to target fs={self.fs} Hz. fs_data must be an integer multiple of self.fs and > self.fs.")

        else:
            raise ValueError(f"Cannot resample from fs_data={fs_data} Hz to target fs={self.fs} Hz. fs_data must be an integer multiple of self.fs and > self.fs.")


    def entire_signal_to_scalogram_images(self, raw_signal, epoch_length=15, overlap=0.5, output_folder='Log/output_scalograms', additional_path='Dataset'):
        """
        Creates a folder of scalogram images starting from a raw signal. Used for dataset preprocessing
        :param raw_signal: a
        :param epoch_length:
        :param overlap:
        :param num_scales:
        :param f_min:
        :param f_max:
        :param w:
        :param output_size:
        :return:
        """

        epoch_samples = int(epoch_length * self.fs)
        hop = int(epoch_samples * (1 - overlap))
        total = len(raw_signal)
        n_epochs = int(np.floor((total-epoch_samples)/hop) + 1)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_folder + "/Plots", exist_ok=True)

        final_signal = nk.eda_clean(raw_signal, sampling_rate=self.fs, method='neurokit')
        img_list_path = []
        #fname = os.path.join(output_folder, f"Plots/Entire.png")
        #plot_signal(raw_signal, fname)
        epoch_base = len(os.listdir(output_folder))
        # Process each epoch
        for idx in range(n_epochs):
            start = idx * hop
            epoch = final_signal[start:start + epoch_samples]
            # Get scalogram image
            image = self.epoch_to_scalogram_image_pywt(epoch)
            # Save on file
            fname = output_folder + "/" + f"epoch_{idx+epoch_base}.png"
            cv2.imwrite(fname, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            fname = additional_path + fname
            img_list_path.append(fname)

        print(f"Saved {n_epochs} scalogram images to '{output_folder}'")

        return img_list_path

    def low_freq_filter(self, raw_signal):
        # Savitzky–Golay smooth at 4Hz
        smoothed = signal.savgol_filter(raw_signal, window_length=9, polyorder=2)

        # Upsample signal
        upsampled = signal.resample_poly(smoothed, up=2, down=1)

        return upsampled

    def plot_scalogram(self, coeff, freqs, time_axis):
        fig, axs = plt.subplots()
        pcm = axs.pcolormesh(time_axis, freqs, coeff, shading='auto')
        axs.set_yscale("log")
        axs.set_xlabel("Time (s)")
        axs.set_ylabel("Frequency (Hz)")
        axs.set_title("Continuous Wavelet Transform (Scaleogram)")
        fig.colorbar(pcm, ax=axs)
        plt.show()


def plot_signal(signal, filename, title=None, xlabel='Sample', ylabel='Amplitude'):
    """
    Plots a 1D signal and saves the plot as an image file.

    :param signal: Iterable of numeric values representing the signal.
    :param filename: Path (including filename) where the plot image will be saved.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    """
    plt.figure()
    plt.plot(signal)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

