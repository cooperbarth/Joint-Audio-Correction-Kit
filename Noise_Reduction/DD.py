#Inspiration from: https://pdfs.semanticscholar.org/8077/e1edd010261398f20719eb1bdc17d19ae678.pdf

import numpy as np, librosa

WINDOW_LENGTH = 2048
HOP_SIZE = 1024
ALPHA = 0.98
START_FRAME = 12

def DD(noisy_signal, alpha=ALPHA, start_frame=START_FRAME):
    """
    Reconstructs the signal by re-adding phase components to the magnitude estimate
    :param noisy_signal: 1D numpy array containing the noisy signal in the time domain
    :param alpha: smoothing constant, defaults to 0.98
    :param start_frame: last frame of sound to consider as noise sample
    :return:
        stft_noisy: stft of original noisy signal
        DD_gains: ndarray containing gain for each bin in the signal
        noise_estimation: 1D numpy array containing average noise up to start_frame
    """

    #make stft of time-domain noisy signal
    stft_noisy = librosa.stft(noisy_signal, win_length=WINDOW_LENGTH, hop_length=HOP_SIZE)

    #calculate average noise over the first n frames of the signal
    noise_estimation = np.mean(np.abs(stft_noisy[:, :start_frame-1]), axis=1)

    #initialization
    filter_gain = np.ones(noise_estimation.shape)
    last_prior_snr = np.zeros(filter_gain.shape)
    last_post_snr = np.zeros(filter_gain.shape)
    num_frames = stft_noisy.shape[1]
    DD_gains = []

    new_sig = np.ones(stft_noisy.shape)

    for frame_number in range(num_frames):
        noisy_frame = np.abs(stft_noisy[:, frame_number])

        #calculate current bin's SNR_post
        post_snr = np.divide(np.square(noisy_frame), noise_estimation)

        #calculate current bin's SNR_prior
        prior_snr = (alpha * last_prior_snr) + (1 - alpha) * post_snr

        #store SNR_post and SNR_prior for next bin
        last_post_snr = post_snr.copy()
        last_prior_snr = prior_snr.copy()

        #calculate the current bin's gain
        filter_gain = np.divide(prior_snr, prior_snr + 1)
        DD_gains.append(filter_gain)
        new_sig[:, frame_number] = filter_gain * new_sig[:, frame_number]

    return stft_noisy, np.asarray(DD_gains).T, noise_estimation