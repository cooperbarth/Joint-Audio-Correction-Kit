import numpy as np, scipy as sp

#Ro should be 0 or another constant when regeneration needed, and 1 otherwise
RO = 0.75

NL = "max"
WINDOW_TYPE = 'hamming'

'''
Inspiration: 
    https://bit.ly/2W6T6dJ
    https://bit.ly/2FfMpz7
    https://bit.ly/2FpNKop
    https://bit.ly/2Cx8eKe
'''
def regeneration(orig_sig, reduced_sig, gain, ro=RO, NL=NL):
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
    S_harmo = np.fft.fft(s_harmo)

    #gamma represents the noise power spectral density of the original noisy signal
    orig_padded = np.concatenate((np.zeros(len(reduced_sig) - len(orig_sig)), orig_sig)) #pad signal
    freqs, gamma = sp.signal.periodogram(orig_padded, window=WINDOW_TYPE, return_onesided=False)

    #calculate SNR_harmo(p, wk) for use in finding suppression gain
    S = np.fft.fft(reduced_sig)
    SNR_harmo = ((ro * (S ** 2)) + ((1 - ro) * (S_harmo ** 2))) / gamma
    
    #calculate suppression gain
    G_harmo = SNR_harmo / (1 + SNR_harmo)

    X = np.fft.fft(orig_padded)
    return np.real(np.fft.ifft(G_harmo * X))

'''
for m in range(max_m):
    begin = m * offset + 1
    iend = m * offset + wl

    nharm = hanwin * newharm[begin:iend]
    ffth = abs(sp.fftpack.fft(nharm))#perform fast fourier transform

    snrham = ((tsnra[:][m + 1]) * (abs(newmags[:][m + 1]) ** 2) + (1 - (tsnra[:][m + 1])) * (ffth ** 2)) / d

    newgain = (snrham / (snrham + 1))
    newgain = gaincontrol(newgain, NFFT / 2)

    newmag = newgain * xmaga(mslice[:], m + 1)

    ffty = newmag * exp(i * phasea(mslice[:], m + 1))

    news(mslice[begin:begin + NFFT - 1]).lvalue = news(mslice[begin:begin + NFFT - 1]) + real(np.fftpack.ifft(ffty, NFFT)) / normFactor
'''