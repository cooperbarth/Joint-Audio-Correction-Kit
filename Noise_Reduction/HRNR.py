import numpy as np
import librosa

WINDOW_LENGTH = 2048
HOP_SIZE = 1024


def HRNR(noisy_stft, TSNR_spectrum, TSNR_gains, noise_estimation, NL="max"):
    """
    Reconstructs the signal by re-adding phase components to the magnitude estimate
    :param noisy_stft: stft of original noisy signal
    :param TSNR_spectrum: clean stft returned by TSNR
    :param TSNR_gains: gains of each stft frame returned by TSNR
    :param noise_estimation: noise estimation average based on first n frames of noisy signal
    :param NL: string representing the non-linear function to be applied to TSNR_spectrum
    :return:
        signal_output: stft of signal after TSNR modification
        TSNR_gains: ndarray containing gain for each bin in signal_output
    """

    # applying non-linear function to TSNR_spectrum
    harmo_spectrum = TSNR_spectrum.copy()
    if NL == "abs":
        harmo_spectrum = np.abs(harmo_spectrum)
    else:
        harmo_spectrum[harmo_spectrum <= 0] = 0.01

    # initialization
    num_frames = TSNR_spectrum.shape[1]
    output_spectrum = np.zeros(TSNR_spectrum.shape, dtype=complex)

    for frame_number in range(num_frames):
        noisy_frame = np.abs(TSNR_spectrum[:, frame_number])
        harmo_frame = np.abs(harmo_spectrum[:, frame_number])
        gain_TSNR = TSNR_gains[:, frame_number]

        # calculate prior SNR
        A = gain_TSNR * (np.abs(noisy_frame) ** 2)
        B = (1 - gain_TSNR) * (np.abs(harmo_frame) ** 2)
        SNR_prior = (A + B) / noise_estimation

        # calculate new gain and apply
        HRNR_gain = np.divide(SNR_prior, SNR_prior + 1)
        output_spectrum[:, frame_number] = noisy_stft[:,
                                                      frame_number] * HRNR_gain

    return librosa.istft(output_spectrum, hop_length=HOP_SIZE, win_length=WINDOW_LENGTH)
