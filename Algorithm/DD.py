import numpy as np

ALPHA = 0.98
START_FRAME = 12

def DD(stft_noisy, alpha=ALPHA, start_frame=START_FRAME):
    noise_estimation = np.mean(np.abs(stft_noisy[:, :start_frame-1]), axis=1)

    filter_gain = np.ones(noise_estimation.shape)
    last_post_snr = filter_gain
    num_frames = stft_noisy.shape[1]

    signal_est_mag = np.zeros(stft_noisy.shape)
    total_gain = []

    for frame_number in range(0, num_frames):
        noisy_frame = np.abs(stft_noisy[:, frame_number])
        current_post_snr = np.divide(np.square(noisy_frame), noise_estimation)
        prior_snr = (alpha * np.square(filter_gain) + last_post_snr) * \
                    (last_post_snr + (1 - alpha) * np.amax(current_post_snr - 1))

        last_post_snr = current_post_snr
        filter_gain = np.divide(prior_snr, prior_snr + 1)
        signal_est_mag[:, frame_number] = np.multiply(filter_gain, noisy_frame)
        total_gain.append(filter_gain)

    return np.asarray(total_gain).T, noise_estimation