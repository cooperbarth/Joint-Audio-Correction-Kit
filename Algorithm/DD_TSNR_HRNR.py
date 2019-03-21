import numpy as np, scipy as sp

#GOD PAPER?: https://pdfs.semanticscholar.org/8077/e1edd010261398f20719eb1bdc17d19ae678.pdf

ALPHA = 0.98
START_FRAME = 12

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

    return np.asarray(signal_gains), noise_estimation

def TSNR(noisy_stft, signal_gains, noise_estimation, alpha=ALPHA):
    """
    Reconstructs the signal by re-adding phase components to the magnitude estimate
    :param signal_stft: stft of original noisy signal
    :param signal_gains: has gains for each frame in stft
    :param noise_estimation: based on first n frames
    :param alpha: smoothing factor, 0.98 recommended
    """

    num_frames = noisy_stft.shape[1]
    signal_output = np.zeros(noisy_stft.shape)

    for frame_number in range(num_frames): #changed this from range(start_frame, end_frame)
        noisy_frame = np.abs(noisy_stft[:, frame_number])

        #Calculating SNR_prior for current frame
        numerator = (signal_gains[frame_number] * noisy_stft[frame_number]) ** 2
        denominator = np.mean(noise_estimation ** 2, axis=1)
        prior_SNR = numerator / denominator

        #Calculating TSNR filter_gain for current frame
        TSNR_gain = np.divide(prior_SNR, prior_SNR + 1)

        signal_output[:, frame_number] = TSNR_gain * noisy_stft[:, frame_number]

    return signal_output