#an algorithm for applying low-pass and SD-rom filters
import numpy as np, scipy as sp, librosa
from scipy.signal import lfilter, butter, freqz
from pysndfx import AudioEffectsChain

#Constants for SD-rom
WINDOW_SIZE = 5
THRESHOLD_1, THRESHOLD_2 = 4, 12
#THRESHOLD_1, THRESHOLD_2, THRESHOLD_3 = 6, 8, 14  ->  for a WINDOW_SIZE of 7

#pass in a window of size n with the center sample under inspection
def SD_rom(window):
    #getting the sample we're looking at
    center_index = len(window) // 2
    center_sample = window[center_index] #x(n)

    #organizing the sliding window i.e. the window without the center sample
    sliding_window = window.copy() #w(n)
    sliding_window = np.concatenate((sliding_window[:center_index], sliding_window[(center_index + 1):]))
    sliding_window = sorted(sliding_window) #r(n)

    #calculating the rank-ordered mean
    ROM = (sliding_window[center_index - 1] + sliding_window[center_index]) / 2

    #construct rank order differences
    rank_order_diff = []
    for i in range(len(sliding_window)):
        if center_sample > ROM:
            rank_order_diff.append(center_sample - sliding_window[-i - 1])
        else:
            rank_order_diff.append(sliding_window[i] - center_sample)
    
    #replace x(n) with ROM if impulse detected
    if rank_order_diff[0] > THRESHOLD_1 or rank_order_diff[1] > THRESHOLD_2: #or rank_order_diff[2] > THRESHOLD_3:
        window[center_index] = ROM
        
    return window

def lowpass(sig, sr, thresh):
    spec_cent = librosa.feature.spectral_centroid(y=sig, sr=sr)
    low_thresh = round(np.median(spec_cent)) * thresh
    rem_noise = AudioEffectsChain().highshelf(gain=-30.0, frequency=low_thresh, slope=0.8)
    return rem_noise(sig)
    
def filter_signal(sig, sr, thresh):
    len_signal, win_start, win_end = len(sig), 0, WINDOW_SIZE

    while win_end < len_signal:
        sig[win_start : win_end] = SD_rom(sig[win_start : win_end])
        win_start += 1
        win_end += 1
    
    return lowpass(sig, sr, thresh)

def wavwrite(filepath, data, sr, norm=True, dtype='int16'):
    if norm:
        data /= np.max(np.abs(data))
    data = data * np.iinfo(dtype).max
    data = data.astype(dtype)
    sp.io.wavfile.write(filepath, sr, data)

audio, sr = librosa.load("trumpet.wav", sr=None)
new_signal = filter_signal(audio, sr, 0.1)
wavwrite("noise_reduced.wav", new_signal, sr)