import numpy as np, scipy as sp, librosa, sys, matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_SR = 44100
WINDOW_TYPE = 'hamming'
WINDOW_LENGTH = 2048
HOP_SIZE = 1024

def generate_noise(signal, snr_desired=5, sr=DEFAULT_SR):
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
    _, signal_est = sp.signal.istft(stft_signal_est, window=WINDOW_TYPE, nperseg=WINDOW_LENGTH,
                                    noverlap=HOP_SIZE)

    return signal_est

#Inspiration from: https://ieeexplore.ieee.org/document/1415074/ and https://bit.ly/2FfMpz7
def regeneration(orig_sig, reduced_sig, ro=0.5, NL="max"):
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
        s_harmo = reduced_sig.copy()
        s_harmo[s_harmo < 0] = 0
    else:
        if NL != "abs":
            print("Non-linear function defaulting to absolute value.")
        s_harmo = np.abs(reduced_sig)
    S_harmo = sp.fftpack.fft(s_harmo)

    #gamma represents the noise power spectral density of the original noisy signal
    orig_padded = np.concatenate((orig_sig, np.zeros(len(reduced_sig) - len(orig_sig)))) #pad signal
    freqs, gamma = sp.signal.periodogram(orig_padded, window=WINDOW_TYPE, return_onesided=False)

    #SNRpost(p, wk) = |X(p, wk)|^2 / gamma(p, wk)
    X = sp.fftpack.fft(orig_padded)
    SNR_post = (X ** 2) / gamma

    #calculate SNR_harmo(p, wk) for use in finding suppression gain
    S = sp.fftpack.fft(reduced_sig)
    SNR_harmo = ((ro * (S ** 2)) + ((1 - ro) * (S_harmo ** 2))) / gamma
    
    #calculate suppression gain
    G_harmo = SNR_harmo / (1 + SNR_harmo)

    #TODO: Is this S or X???
    return sp.fftpack.ifft(G_harmo * S)

def wavwrite(filepath, data, sr, norm=True, dtype='int16'):
    if norm:
        data /= np.max(np.abs(data))
    data = data * np.iinfo(dtype).max
    data = data.astype(dtype)
    sp.io.wavfile.write(filepath, sr, data)


def wiener_filtering(clean_signal, filename):
    """
    Performs Wiener Filtering on a file located at filepath
    :param clean_signal: 1D numpy array containing the signal of a clean audio file
    :param filename: string of the audio file name
    """
    noisy_signal = generate_noise(clean_signal)

    write_name = filename.split(".")[0]
    new_path = "test_audio_noisy/" + write_name + "_noisy.wav"
    wavwrite(new_path, noisy_signal, DEFAULT_SR)

    _, _, stft_noisy = sp.signal.stft(noisy_signal, window=WINDOW_TYPE, nperseg=WINDOW_LENGTH,
                                      noverlap=HOP_SIZE)

    signal_est_mag = denoising(stft_noisy)

    signal_est_reconstruction = reconstruction(stft_noisy, signal_est_mag)
    new_path = "test_audio_reconstructed/" + write_name + "_reconstructed.wav"
    wavwrite(new_path, noisy_signal, DEFAULT_SR)

    signal_est = regeneration(noisy_signal, signal_est_reconstruction)

    new_path = "test_audio_results/" + write_name + "_reduced.wav"
    wavwrite(new_path, np.abs(signal_est), DEFAULT_SR)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            if filename[-4:] != ".wav":
                filename += ".wav"
            filepath = "test_audio/" + filename
            try:
                clean_signal, _ = librosa.load(filepath, sr=DEFAULT_SR)
            except:
                print(filename + " is not a valid file name.")
                continue
            wiener_filtering(clean_signal, filename)
    else:
        pathlist = Path("test_audio").glob('**/*.wav')
        for filepath in pathlist:
            clean_signal, _ = librosa.load(filepath, sr=DEFAULT_SR)
            wiener_filtering(clean_signal, str(filepath).split("/")[1])
