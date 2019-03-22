
import numpy as np
import librosa
from scipy import fftpack as fp

WINDOW_LENGTH = 2048
HOP_SIZE = 1024


def snr(reduced_signal, original_signal):
    stft_original = librosa.stft(original_signal, win_length=WINDOW_LENGTH, hop_length=HOP_SIZE)
    stft_reduced = librosa.stft(reduced_signal, win_length=WINDOW_LENGTH, hop_length=HOP_SIZE)

    seg_snr = 0
    for frame in range(stft_original.shape[1]):
        original_slice = fp.ifft(stft_original[:, frame])
        reduced_slice = fp.ifft(stft_reduced[:, frame])
        numerator = np.sum(np.square(np.abs(original_slice)))
        denominator = np.sum(np.square(np.abs(reduced_slice) - np.abs(original_slice)))
        seg_snr += numerator / denominator

    seg_snr = (1 / stft_original.shape[1]) * seg_snr
    print("Average Segment SNR: " + str(seg_snr))
    return
