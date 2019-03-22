import numpy as np, librosa
from pysndfx import AudioEffectsChain

SLOPE = 0.9
THRESHOLD = 10


def highpass(sig, sr, high_thresh=THRESHOLD):
    """
    Passes signal through a highpass filter
    :param sig: signal to be highpassed
    :param sr: sample rate of the signal
    :param high_thresh: Threshold above which frequencies should be highpassed out
    :return:
        1d numpy array containing the filtered signal
    """

    spec_cent = librosa.feature.spectral_centroid(y=sig, sr=sr)
    spec_med = round(np.median(spec_cent))

    high_thresh *= spec_med
    if high_thresh > sr/2:
        high_thresh = sr/2
    rem_noise = AudioEffectsChain().highshelf(frequency=high_thresh, slope=SLOPE)

    return rem_noise(sig)
