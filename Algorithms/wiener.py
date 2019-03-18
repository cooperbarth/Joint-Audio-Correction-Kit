import numpy as np
import scipy as sp
import librosa
import sys
from pathlib import Path

default_sr = 44100
window_type = 'hamming'
window_length = 2048
hop_size = 1024


def generate_noise(signal, snr_desired=5, sr=44100):
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


def denoising(stft_noisy, alpha=0.95, start_frame=12):
    """
    Takes in a fourier transformed signal and noise sample from the signal and reduces the noise in the signal
    :param stft_noisy: slice from original signal that functions as the noise
    :param alpha: weighting of cumulative snr when adding new snr
    :param start_frame: frame of stft_signal where the noise ends
    :return: a 2-d numpy array of the magnitude of the noise reduced signal (fourier without phase)
    """
    noise_estimation = np.mean(np.abs(stft_noisy[:, :start_frame-1]), axis=1)

    filter_gain = np.ones(noise_estimation.shape)
    last_post_snr = filter_gain
    num_frames = stft_noisy.shape[1]

    signal_est_mag = np.zeros(stft_noisy.shape)

    for frame_number in range(start_frame, num_frames):
        noisy_frame = np.abs(stft_noisy[:, frame_number])
        current_post_snr = np.divide(np.square(noisy_frame), noise_estimation)
        prior_snr = (alpha * np.square(filter_gain) + last_post_snr) * \
                    (last_post_snr + (1 - alpha) * np.amax(current_post_snr - 1))

        last_post_snr = current_post_snr
        filter_gain = np.divide(prior_snr, prior_snr + 1)
        signal_est_mag[:, frame_number] = np.multiply(filter_gain, noisy_frame)

    return signal_est_mag


def reconstruction(stft_noisy, signal_est_mag):
    """
    Reconstructs the signal by re-adding phase components to the magnitude estimate
    :param stft_noisy: the fourier transform of a noisy signal
    :param signal_est_mag: the noise reduced signal's estimated magnitude
    :return: Noise reduced signal in the time domain
    """
    phase_noisy = np.angle(stft_noisy)
    stft_signal_est = np.multiply(signal_est_mag, np.exp(1j * phase_noisy))
    _, signal_est = sp.signal.istft(stft_signal_est, window=window_type, nperseg=window_length,
                                    noverlap=hop_size)

    return signal_est

#Inspiration from: https://ieeexplore.ieee.org/document/1415074/
def regeneration(orig_sig, reduced_sig, ro, NL="max"):
    """
    Reincludes lost harmonics into a reconstructed signal.
    :param orig_sig: the original noisy signal in the time domain
    :param reduced_sig: the signal after reconstuction in the time domain
    :param ro: the mixing level, typically a double between 0 and 1
    :param NL: the non-linear function to be used in generating the harmonics
    :return: 1D numpy array with harmonic-added signal
    """

    #getting S_harmo, the harmonic amplifier in the frequency domain
    if NL == "max":
        s_harmo = orig_sig.copy()
        s_harmo[s_harmo < 0] = 0
    else:
        if NL != "abs":
            print("Non-linear function defaulting to absolute value.")
        s_harmo = np.abs(orig_sig)
    S_harmo = sp.fftpack.fft(s_harmo)

    #gamma represents the noise power spectral density of the original noisy signal
    orig_sig_freq = sp.fftpack.fft(orig_sig)
    gamma = sp.signal.periodogram(orig_sig_freq, fs=default_sr, window=window_length)
    
    #SNRpost(p, wk) = |X(p, wk)|^2 / gamma(p, wk)
    SNR_post = (orig_sig_freq ** 2) / gamma

    #TODO: this is probably wrong
    #defining our filtering function
    def h(harmo, post):
        return sp.signal.wiener(harmo)

    

def wavwrite(filepath, data, sr, norm=True, dtype='int16'):
    if norm:
        data /= np.max(np.abs(data))
    data = data * np.iinfo(dtype).max
    data = data.astype(dtype)
    sp.io.wavfile.write(filepath, sr, data)


def wiener_filtering(filepath):
    """
    Performs Wiener Filtering on a file located at filepath
    :param filepath: The location of the source file to have noise applied and then removed
    :return:
    """
    clean_signal, _ = librosa.load(filepath, sr=default_sr)
    noisy_signal = generate_noise(clean_signal)

    write_name = str(path).split("/")[1].split(".")[0]
    new_path = "test_audio_noisy/" + write_name + "_noisy.wav"
    wavwrite(new_path, noisy_signal, default_sr)

    _, _, stft_noisy = sp.signal.stft(noisy_signal, window=window_type, nperseg=window_length,
                                      noverlap=hop_size)

    signal_est_mag = denoising(stft_noisy, start_frame=12)
    signal_est = reconstruction(stft_noisy, signal_est_mag)

    write_name = str(path).split("/")[1].split(".")[0]
    new_path = "test_audio_results/" + write_name + "_reduced.wav"
    wavwrite(new_path, signal_est, default_sr)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            if filename[-4:] != ".wav":
                filename += ".wav"
            try:
                wiener_filtering("test_audio/" + filename)
            except:
                print(filename + " is not a valid file name.")
    else:
        pathlist = Path("test_audio").glob('**/*.wav')
        for path in pathlist:
            wiener_filtering(path)
