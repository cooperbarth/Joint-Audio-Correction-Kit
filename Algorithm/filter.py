import numpy as np, scipy as sp, matplotlib.pyplot as plt, sys, librosa
from pathlib import Path
from librosa import load

from generate_noise import generate_noise

sys.path.append('utility')
from wavwrite import wavwrite

from DD_TSNR_HRNR import DD
from DD_TSNR_HRNR import TSNR
from DD_TSNR_HRNR import HRNR

np.seterr(divide='ignore', invalid='ignore')

DEFAULT_SR = 44100
WINDOW_TYPE = 'hamming'
WINDOW_LENGTH = 2048
HOP_SIZE = 1024

def wiener_filtering(clean_signal, filename):
    """
    Performs Wiener Filtering on a file located at filepath
    :param clean_signal: 1D numpy array containing the signal of a clean audio file
    :param filename: string of the audio file name
    """
    if len(clean_signal) > 400000:
        clean_signal = clean_signal[:400000]

    write_name = filename.split(".")[0]

    if '+' not in filename:
        noisy_signal = generate_noise(clean_signal)

        new_path = "audio/test_audio_noisy/" + write_name + "_noisy.wav"
        wavwrite(new_path, noisy_signal, DEFAULT_SR)
    else:
        noisy_signal = clean_signal.copy()

    stft_noisy = librosa.stft(noisy_signal, win_length=WINDOW_LENGTH, hop_length=HOP_SIZE)

    DD_gains, noise_est = DD(stft_noisy)
    TSNR_sig, TSNR_gains = TSNR(stft_noisy, DD_gains, noise_est)
    signal_est = HRNR(stft_noisy, TSNR_sig, TSNR_gains, noise_est)

    new_path = "audio/test_audio_results/" + write_name + "_reduced.wav"
    wavwrite(new_path, signal_est, DEFAULT_SR)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            if filename[-4:] != ".wav":
                filename += ".wav"
            filepath = "audio/test_audio/" + filename
            try:
                clean_signal, _ = load(filepath, sr=DEFAULT_SR)
            except:
                print(filename + " is not a valid file name.")
                continue
            wiener_filtering(clean_signal, filename)
    else:
        pathlist = Path("audio/test_audio").glob('**/*.wav')
        for filepath in pathlist:
            clean_signal, _ = load(filepath, sr=DEFAULT_SR)
            wiener_filtering(clean_signal, str(filepath).split("/")[1])