import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import config


def plot_audio(x, sr, filename="tmp", figsize=(16, 4)):
    length = float(x.shape[0]) / sr
    t = np.linspace(0, length, x.shape[0])
    matplotlib.use('agg')
    plt.figure(figsize=figsize)
    plt.plot(t, x)
    plt.title(filename)
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    path = config.API_TEMP_DIRECTORY + filename + '.png'
    plt.savefig(path)
    return path
