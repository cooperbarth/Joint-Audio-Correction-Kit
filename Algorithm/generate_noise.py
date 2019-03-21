import numpy as np

SNR = 20
DEFAULT_SR = 44100

def generate_noise(signal, snr_desired=SNR, sr=DEFAULT_SR):
    """
    Applies background noise to a clean track
    :param signal: original signal that will have noise applied to it
    :param snr_desired: sample-to-noise ratio desired (in dB)
    :param sr: the sample rate of the noise (44100)
    :return: a 1-d numpy array of a noisy signal made from a sinusoid of frequency fs, duration signal_duration
    """
    signal_length = signal.size
    noise_length = int(signal_length + np.ceil(0.5 * sr))

    # Apply the noise factor to the noisy signal
    noise_factor = np.sqrt((1/signal_length) * (np.sum(np.square(signal)) / np.power(10, snr_desired / 10)))
    noisy_signal = noise_factor * np.random.randn(noise_length)

    signal_start = int(np.floor(0.5*sr))
    signal_range = signal_start + np.arange(0, signal_length)

    noisy_signal[signal_range] = noisy_signal[signal_range] + signal

    return noisy_signal