import numpy as np

def TSNR(noisy_stft, signal_gains, noise_estimation):
    """
    Reconstructs the signal by re-adding phase components to the magnitude estimate
    :param signal_stft: stft of original noisy signal
    :param signal_gains: has gains for each frame in stft
    :param noise_estimation: based on first n frames
    """

    num_frames = noisy_stft.shape[1]
    signal_output = np.zeros(noisy_stft.shape, dtype=complex)
    TSNR_gains = []

    for frame_number in range(num_frames):
        noisy_frame = np.abs(noisy_stft[:, frame_number])

        #Calculating SNR_prior for current frame
        numerator = (signal_gains[:, frame_number] * noisy_frame) ** 2
        prior_SNR = numerator / noise_estimation

        #Calculating TSNR filter_gain for current frame
        TSNR_gain = np.divide(prior_SNR, prior_SNR + 1)
        TSNR_gains.append(TSNR_gain)

        signal_output[:, frame_number] = TSNR_gain * noisy_stft[:, frame_number]

    return signal_output, np.asarray(TSNR_gains).T