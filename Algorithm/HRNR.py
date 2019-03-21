import numpy as np, librosa

WINDOW_LENGTH = 2048
HOP_SIZE = 1024

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