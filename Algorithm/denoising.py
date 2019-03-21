import numpy as np

ALPHA = 0.95
START_FRAME = 12

def denoising(stft_noisy, alpha=ALPHA, start_frame=START_FRAME):
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
    total_gain = []

    for frame_number in range(0, num_frames): #changed this from range(start_frame, end_frame)
        noisy_frame = np.abs(stft_noisy[:, frame_number])
        current_post_snr = np.divide(np.square(noisy_frame), noise_estimation)
        prior_snr = (alpha * np.square(filter_gain) + last_post_snr) * \
                    (last_post_snr + (1 - alpha) * np.amax(current_post_snr - 1))

        last_post_snr = current_post_snr
        filter_gain = np.divide(prior_snr, prior_snr + 1)
        signal_est_mag[:, frame_number] = np.multiply(filter_gain, noisy_frame)

    return signal_est_mag, np.asarray(total_gain)