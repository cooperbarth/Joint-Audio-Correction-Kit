import numpy as np
from librosa import load
from pathlib import Path
from wavwrite import wavwrite


def merge_audio(clean_audio, noise_audio, audio_sr, snr, clean_location, noise_location):
    resized_noise = np.resize(noise_audio, len(clean_audio))
    combined_audio = clean_audio + (resized_noise * 1/snr)

    clean_name = clean_location.split('.')[2].split('/')[4]
    noise_name = noise_location.split('.')[2].split('/')[4]
    new_path = "../audio/merging/test_merged/" + clean_name + "+" + noise_name + ".wav"
    wavwrite(new_path, combined_audio, audio_sr)
    return


if __name__ == "__main__":
    clean_path = Path("../audio/merging/test_clean").glob('**/*.wav')
    for clean in clean_path:
        clean_signal, clean_sr = load(str(clean), sr=None)
        noise_path = Path("../audio/merging/test_noise").glob('**/*.wav')
        for noise in noise_path:
            noise_signal, _ = load(str(noise), sr=clean_sr)
            merge_audio(clean_signal, noise_signal, clean_sr, 15, str(clean), str(noise))
