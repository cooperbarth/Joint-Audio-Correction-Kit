import numpy as np, scipy as sp, librosa

#GOD PAPER?: https://pdfs.semanticscholar.org/8077/e1edd010261398f20719eb1bdc17d19ae678.pdf

ALPHA = 0.98
START_FRAME = 12

DEFAULT_SR = 44100
WINDOW_TYPE = 'hamming'
WINDOW_LENGTH = 2048
HOP_SIZE = 1024

def DD(stft_noisy, alpha=ALPHA, start_frame=START_FRAME):
    noise_estimation = np.mean(np.abs(stft_noisy[:, :start_frame-1]), axis=1)

    filter_gain = np.ones(noise_estimation.shape)
    last_post_snr = filter_gain
    num_frames = stft_noisy.shape[1]

    signal_est_mag = np.zeros(stft_noisy.shape)
    signal_gains = []

    for frame_number in range(0, num_frames): #changed this from range(start_frame, end_frame)
        noisy_frame = np.abs(stft_noisy[:, frame_number])
        current_post_snr = np.divide(np.square(noisy_frame), noise_estimation)
        prior_snr = (alpha * np.square(filter_gain) + last_post_snr) * \
                    (last_post_snr + (1 - alpha) * np.amax(current_post_snr - 1))
        last_post_snr = current_post_snr

        current_gain = np.divide(prior_snr, prior_snr + 1)
        filter_gain = current_gain
        signal_gains.append(current_gain)

    return np.asarray(signal_gains).T, noise_estimation

def TSNR(noisy_stft, signal_gains, noise_estimation, alpha=ALPHA):
    """
    Reconstructs the signal by re-adding phase components to the magnitude estimate
    :param signal_stft: stft of original noisy signal
    :param signal_gains: has gains for each frame in stft
    :param noise_estimation: based on first n frames
    :param alpha: smoothing factor, 0.98 recommended
    """

    num_frames = noisy_stft.shape[1]
    signal_output = np.zeros(noisy_stft.shape, dtype=complex)
    TSNR_gains = []

    for frame_number in range(num_frames): #changed this from range(start_frame, end_frame)
        noisy_frame = np.abs(noisy_stft[:, frame_number])

        #Calculating SNR_prior for current frame
        numerator = (signal_gains[:, frame_number] * noisy_stft[:, frame_number]) ** 2
        prior_SNR = numerator / noise_estimation

        #Calculating TSNR filter_gain for current frame
        TSNR_gain = np.divide(prior_SNR, prior_SNR + 1)
        TSNR_gains.append(TSNR_gain)

        signal_output[:, frame_number] = TSNR_gain * noisy_stft[:, frame_number]

    return signal_output, np.asarray(TSNR_gains).T

def HRNR(noisy_stft, speech_spectrum, TSNR_gains, noise_estimation):
    harmo_spectrum = speech_spectrum.copy()
    harmo_spectrum[harmo_spectrum < 0] = 0

    num_frames = speech_spectrum.shape[1]

    output_spectrum = np.zeros(speech_spectrum.shape, dtype=complex)

    for frame_number in range(num_frames):
        noisy_frame = np.abs(speech_spectrum[:, frame_number])
        harmo_frame = np.abs(harmo_spectrum[:, frame_number])
        gain_TSNR = TSNR_gains[:, frame_number]

        A = gain_TSNR * (np.abs(noisy_frame) ** 2)
        B = (1 - gain_TSNR) * (np.abs(harmo_frame) ** 2)

        SNR_prior = (A + B) / noise_estimation
        HRNR_gain = np.divide(SNR_prior, SNR_prior + 1)

        output_spectrum[:, frame_number] = noisy_stft[:, frame_number] * HRNR_gain

    return librosa.istft(output_spectrum, hop_length=HOP_SIZE, win_length=WINDOW_LENGTH)