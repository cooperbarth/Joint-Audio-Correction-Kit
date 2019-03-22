import numpy as np, scipy as sp, librosa, matplotlib, matplotlib.pyplot as plt

DEFAULT_SR = 44100
WINDOW_LENGTH = 2048
HOP_SIZE = 1024

def plot_audio(x, sr, filename="tmp", figsize=(16, 4)):
    length = float(x.shape[0]) / sr
    t = np.linspace(0, length, x.shape[0])
    matplotlib.use('agg')
    
    plt.figure(figsize=figsize)
    plt.plot(t, x)
    plt.title(filename)
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.savefig("graphs/" + filename + ".png")

def plt_spectrogram(X, win_length=WINDOW_LENGTH, hop_size=HOP_SIZE, sample_rate=DEFAULT_SR, zoom_x=None, zoom_y=None, tick_labels='time-freq', filename="tmp", figsize=(16, 4)):
    matplotlib.use('agg')
    X = librosa.stft(X, win_length, hop_size)
    Nf, Nt = np.shape(X)
    X = 20 * np.log10(np.abs(X))
    X = X[0:int(Nf / 2) + 1]
    Nf = np.shape(X)[0]
    times = (hop_size / float(sample_rate)) * np.arange(Nt)
    freqs = (float(sample_rate) / win_length) * np.arange(Nf)
    times_matrix, freqs_matrix = np.meshgrid(times, freqs)
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
    if zoom_x is None and zoom_y is None:
        plt.axis('tight')
    if zoom_x is not None:
        plt.xlim(zoom_x)
    if zoom_y is not None:
        plt.ylim(zoom_y)
    
    path = "graphs/" + filename + ".png"
    plt.savefig(path)