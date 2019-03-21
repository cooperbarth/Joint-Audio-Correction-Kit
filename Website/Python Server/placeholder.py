import numpy as np, sys
from librosa import load, effects

np.seterr(divide='ignore', invalid='ignore')
sys.path.append('../../Noise_Reduction')
from DD import DD
from TSNR import TSNR
from HRNR import HRNR
from highpass import highpass

MAX_SIGNAL_LENGTH = 400000

def this_is_going_to_totally_work_right(signal, sample_rate):
    if len(signal) > MAX_SIGNAL_LENGTH:
        signal = signal[:MAX_SIGNAL_LENGTH]

    stft_noisy, DD_gains, noise_est = DD(signal)
    TSNR_sig, TSNR_gains = TSNR(stft_noisy, DD_gains, noise_est)
    new_signal = HRNR(stft_noisy, TSNR_sig, TSNR_gains, noise_est)
    new_signal = highpass(new_signal, sample_rate)

    return new_signal
