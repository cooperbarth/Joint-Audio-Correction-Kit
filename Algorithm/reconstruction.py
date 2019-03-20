import numpy as np
from scipy.signal import istft

WINDOW_TYPE = 'hamming'
WINDOW_LENGTH = 2048
HOP_SIZE = 1024

def reconstruction(stft_noisy, signal_est_mag):
    """
    Reconstructs the signal by re-adding phase components to the magnitude estimate
    :param stft_noisy: the fourier transform of a noisy signal
    :param signal_est_mag: the noise reduced signal's estimated magnitude
    :return: Noise reduced signal in the time domain
    """
    phase_noisy = np.angle(stft_noisy)
    stft_signal_est = np.multiply(signal_est_mag, np.exp(1j * phase_noisy))
    _, signal_est = istft(stft_signal_est, window=WINDOW_TYPE, nperseg=WINDOW_LENGTH,
                                    noverlap=HOP_SIZE)

    return signal_est