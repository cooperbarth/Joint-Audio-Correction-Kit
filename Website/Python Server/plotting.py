import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import config
from librosa import stft


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


def plt_spectrogram(X, win_length, hop_size, sample_rate, zoom_x=None, zoom_y=None, tick_labels='time-freq', filename="tmp", figsize=(10, 5)):
    """
    Plots the log magnitude spectrogram.

    Input Parameters:
    ------------------
    X: The signal.
    win_length: the length of the analysis window
    hop_size: the hop size between adjacent windows
    sample_rate: sampling frequency
    tick_labels: the type of x and y tick labels, there are two options:
                 'time-freq': shows times (sec) on the x-axis and frequency (Hz) on the y-axis (default)
                 'bin-frame': shows time frame numbers on the x-axis and frequency bin numbers on the y-axis

    zoom_x: 1 by 2 numpy array containing the range of values on the x-axis, e.g. zoom_t = np.array([x_start,x_end])
    zoom_y: 1 by 2 numpy array containing the range of values on the y-axis, e.g. zoom_f = np.array([y_start,y_end])


    Returns:
    ---------
    times: 1D real numpy array containing time instances corresponding to stft frames
    freqs: 1D real numpy array containing frequencies of analyasis up to Nyquist rate
    2D plot of the magnitude spectrogram
    """

    matplotlib.use('agg')

    # Find the size of stft
    X = stft(X, win_length, hop_size)

    Nf, Nt = np.shape(X)

    # Compute the log magnitude spectrogram
    X = 20 * np.log10(np.abs(X))

    # Extract the lower half of the spectrum for each time frame
    # make sure to include both 0 and Nyquist frequency
    X = X[0:int(Nf / 2) + 1]
    Nf = np.shape(X)[0]

    # Generate time vector for plotting
    times = (hop_size / float(sample_rate)) * np.arange(Nt)

    # Generate frequency vector for plotting
    freqs = (float(sample_rate) / win_length) * np.arange(Nf)

    # Generate time and frequency matrices for pcolormesh
    times_matrix, freqs_matrix = np.meshgrid(times, freqs)

    # Plot the log magnitude spectrogram
    plt.figure(figsize=figsize)
    plt.title('Log magnitude spectrogram')
    if tick_labels == 'bin-frame':
        plt.pcolormesh(X)
        plt.xlabel('Time-frame Number')
        plt.ylabel('Frequency-bin Number')
    else:
        plt.pcolormesh(times_matrix, freqs_matrix, X)
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')

    # Zoom in on the plot if specified
    if zoom_x is None and zoom_y is None:
        plt.axis('tight')

    if zoom_x is not None:
        plt.xlim(zoom_x)

    if zoom_y is not None:
        plt.ylim(zoom_y)

    path = config.API_TEMP_DIRECTORY + filename + '.png'
    plt.savefig(path)
    return path
