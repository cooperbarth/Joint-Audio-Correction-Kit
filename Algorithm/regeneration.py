import numpy as np, scipy as sp

#Ro should be 0 or another constant when regeneration needed, and 1 otherwise
RO = 0.1

NL = "max"
WINDOW_TYPE = 'hamming'

'''
Inspiration: 
    https://bit.ly/2W6T6dJ
    https://bit.ly/2FfMpz7
    https://bit.ly/2FpNKop
    https://bit.ly/2Cx8eKe - MATLAB code
'''
def regeneration(orig_sig, reduced_sig, ro=RO, NL=NL):
    """
    Reincludes lost harmonics into a reconstructed signal.
    :param orig_sig: the original noisy signal in the time domain
    :param reduced_sig: the signal after reconstuction in the time domain
    :param ro: the mixing level, typically a double between 0 and 1
    :param NL: the non-linear function to be used in generating the harmonics
    :return: 1D numpy array with harmonic-added signal
    """

    #getting S_harmo, the harmonic amplifier in the frequency domain
    if NL == "max":
        s_harmo = reduced_sig.copy()
        s_harmo[s_harmo < 0] = 0
    else:
        if NL != "abs":
            print("Non-linear function defaulting to absolute value.")
        s_harmo = np.abs(reduced_sig)
    S_harmo = sp.fftpack.fft(s_harmo)

    #gamma represents the noise power spectral density of the original noisy signal
    orig_padded = np.concatenate((np.zeros(len(reduced_sig) - len(orig_sig)), orig_sig)) #pad signal
    freqs, gamma = sp.signal.periodogram(orig_padded, window=WINDOW_TYPE, return_onesided=False)

    #SNRpost(p, wk) = |X(p, wk)|^2 / gamma(p, wk)
    X = sp.fftpack.fft(orig_padded)
    SNR_post = (X ** 2) / gamma

    #calculate SNR_harmo(p, wk) for use in finding suppression gain
    S = sp.fftpack.fft(reduced_sig)
    SNR_harmo = ((ro * (S ** 2)) + ((1 - ro) * (S_harmo ** 2))) / gamma
    
    #calculate suppression gain
    G_harmo = SNR_harmo / (1 + SNR_harmo)

    return np.abs(sp.fftpack.ifft(G_harmo * X))