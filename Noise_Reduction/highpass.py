import numpy as np, librosa
from pysndfx import AudioEffectsChain

SLOPE = 0.9
THRESHOLD = 10

def highpass(sig, sr, high_thresh=THRESHOLD):
    spec_cent = librosa.feature.spectral_centroid(y=sig, sr=sr)
    spec_med = round(np.median(spec_cent))

    low_thresh = (1 - high_thresh) * spec_med
    high_thresh *= spec_med
    if high_thresh > sr/2:
        high_thresh = sr/2
    rem_noise = AudioEffectsChain().highshelf(frequency=high_thresh, slope=SLOPE)

    return rem_noise(sig)