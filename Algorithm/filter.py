import numpy as np, scipy as sp, librosa, sys, matplotlib, matplotlib.pyplot as plt
from pathlib import Path

from generate_noise import generate_noise

np.seterr(divide='ignore', invalid='ignore')

DEFAULT_SR = 44100
WINDOW_TYPE = 'hamming'
WINDOW_LENGTH = 2048
HOP_SIZE = 1024

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

    for frame_number in range(0, num_frames): #changed this from range(start_frame, end_frame)
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

def plt_spectrogram(X, win_length, hop_size, sample_rate, zoom_x=None, zoom_y=None, tick_labels='time-freq', filename="tmp", figsize=(16, 4)):
    matplotlib.use('agg')
    X = librosa.stft(X, win_length, hop_size)
    Nf, Nt = np.shape(X)
    X = 20 * np.log10(np.abs(X))
    X = X[0:int(Nf / 2) + 1]
    Nf = np.shape(X)[0]
    times = (hop_size / float(sample_rate)) * np.arange(Nt)
    freqs = (float(sample_rate) / win_length) * np.arange(Nf)
    times_matrix, freqs_matrix = np.meshgrid(times, freqs)
    plt.figure(figsize=figsize)
    plt.title('Log magnitude spectrogram')
    if tick_labels == 'bin-frame':
        plt.pcolormesh(X)
        plt.xlabel('Time-frame Number')
        plt.ylabel('Frequency-bin Number')
    else:
        plt.pcolormesh(times_matrix, freqs_matrix, X)
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
    if zoom_x is None and zoom_y is None:
        plt.axis('tight')
    if zoom_x is not None:
        plt.xlim(zoom_x)
    if zoom_y is not None:
        plt.ylim(zoom_y)
    
    path = "Spectrograms/" + filename + ".png"
    plt.savefig(path)

#Inspiration from: https://ieeexplore.ieee.org/document/1415074/ and https://bit.ly/2FfMpz7
def regeneration(orig_sig, reduced_sig, ro=0.1, NL="max"):
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
    orig_padded = np.concatenate((np.zeros(len(reduced_sig) - len(orig_sig)), orig_sig)) #pad signal
    freqs, gamma = sp.signal.periodogram(orig_padded, window=WINDOW_TYPE, return_onesided=False)

    #SNRpost(p, wk) = |X(p, wk)|^2 / gamma(p, wk)
    X = sp.fftpack.fft(orig_padded)
    SNR_post = (X ** 2) / gamma

    #calculate SNR_harmo(p, wk) for use in finding suppression gain
    S = sp.fftpack.fft(reduced_sig)
    SNR_harmo = ((ro * (S ** 2)) + ((1 - ro) * (S_harmo ** 2))) / gamma
    
    #calculate suppression gain
    G_harmo = SNR_harmo / (1 + SNR_harmo)

    return np.abs(sp.fftpack.ifft(G_harmo * X))

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
    if len(clean_signal) > 400000:
        clean_signal = clean_signal[:400000]

    plt_spectrogram(clean_signal, WINDOW_LENGTH, HOP_SIZE, DEFAULT_SR, filename='clean')

    noisy_signal = generate_noise(clean_signal)
    plt_spectrogram(noisy_signal, WINDOW_LENGTH, HOP_SIZE, DEFAULT_SR, filename='noisy')

    write_name = filename.split(".")[0]
    new_path = "audio/test_audio_noisy/" + write_name + "_noisy.wav"
    wavwrite(new_path, noisy_signal, DEFAULT_SR)

    _, _, stft_noisy = sp.signal.stft(noisy_signal, window=WINDOW_TYPE, nperseg=WINDOW_LENGTH,
                                      noverlap=HOP_SIZE)

    signal_est_mag = denoising(stft_noisy)

    signal_est_reconstruction = reconstruction(stft_noisy, signal_est_mag)
    new_path = "audio/test_audio_reconstructed/" + write_name + "_reconstructed.wav"
    wavwrite(new_path, signal_est_reconstruction, DEFAULT_SR)
    plt_spectrogram(signal_est_reconstruction, WINDOW_LENGTH, HOP_SIZE, DEFAULT_SR, filename='reconstructed')

    signal_est = regeneration(noisy_signal, signal_est_reconstruction)
    plt_spectrogram(signal_est, WINDOW_LENGTH, HOP_SIZE, DEFAULT_SR, filename='regenerated')

    new_path = "audio/test_audio_results/" + write_name + "_reduced.wav"
    wavwrite(new_path, np.abs(signal_est), DEFAULT_SR)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            if filename[-4:] != ".wav":
                filename += ".wav"
            filepath = "audio/test_audio/" + filename
            try:
                clean_signal, _ = librosa.load(filepath, sr=DEFAULT_SR)
            except:
                print(filename + " is not a valid file name.")
                continue
            wiener_filtering(clean_signal, filename)
    else:
        pathlist = Path("audio/test_audio").glob('**/*.wav')
        for filepath in pathlist:
            clean_signal, _ = librosa.load(filepath, sr=DEFAULT_SR)
            wiener_filtering(clean_signal, str(filepath).split("/")[1])