import numpy as np
import sys
from librosa import load, effects

np.seterr(divide='ignore', invalid='ignore')
sys.path.append('../../Noise_Reduction')
from DD import DD
from TSNR import TSNR
from HRNR import HRNR
from highpass import highpass

def noise_filter(signal, sample_rate):

    stft_noisy, DD_gains, noise_est = DD(signal)
    TSNR_sig, TSNR_gains = TSNR(stft_noisy, DD_gains, noise_est)
    new_signal = HRNR(stft_noisy, TSNR_sig, TSNR_gains, noise_est)
    new_signal = highpass(new_signal, sample_rate)

    return new_signal
