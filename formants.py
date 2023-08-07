import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import soundfile as sf

"""
Implementation of the CheapTrick spectral smoothing algorithm used for formant extraction.

[1] Morise, Masanori. “CheapTrick, a Spectral Envelope Estimator for High-Quality Speech 
    Synthesis.” Speech Communication 67 (March 2015): 1-7. 
    https://doi.org/10.1016/j.specom.2014.09.003.

"""


def gaussian(x, mu, sig):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)

def normalize(x):
    m = np.max(np.abs(x))
    if m > 0:
        return x / m
    else:
        return x

def source_filter_model(t, f0=130, formants=[400, 800]):
    sample_rate = 1/(t[1] - t[0])
    # source 
    source = np.zeros(num_samples)
    fk = f0
    while fk < sample_rate/2:
        source += np.cos(2.0*np.pi*fk * t)
        fk += f0

    source *= sig.windows.hann(num_samples)

    # filter
    fft_size = (num_samples*2)-1
    f = np.linspace(-sample_rate/2, sample_rate/2, fft_size, False)
    filter_spectrum = np.zeros(fft_size)
    for formant in formants:
        filter_spectrum += gaussian(f, formant, 100)
    source_spectrum = np.fft.fft(source, fft_size)
    source_spectrum = np.fft.fftshift(source_spectrum)
    mixture_spectrum = filter_spectrum * source_spectrum
    mixture_spectrum = np.fft.fftshift(mixture_spectrum)
    mixture = np.real(np.fft.ifft(mixture_spectrum))
    mixture = mixture[:len(t)]

    return mixture 

sample_rate = 16000
dur = 0.3
num_samples = int(dur * sample_rate)
t = np.linspace(0, dur, num_samples, False)

# source audio
f0 = 130
formants = [400]
signal = source_filter_model(t, f0=f0, formants=formants)

# analysis 
T0 = 1/f0

q0 = 1.18
q1 = -0.09

fft_size = len(signal)
signal_spectrum = np.fft.fft(signal, fft_size)
signal_spectrum = signal_spectrum[:(len(signal_spectrum)//2)]
signal_power_spectrum = np.power(signal_spectrum, 2)
signal_cepstrum = np.fft.ifft(np.log(signal_power_spectrum))
quefrency = np.linspace(0, dur, len(signal_cepstrum), False)
# quefrency = np.linspace(-dur/2, dur/2, num_samples, False)
sinc_lifter = np.zeros(len(quefrency))

j = 0
for tau in quefrency:
    if tau != 0:
        sinc_lifter[j] = np.sin(np.pi * f0 * tau) / (np.pi * f0 * tau)
    else:
        sinc_lifter[j] = 0
    j+=1

cos_lifter = q0 + 2.0*q1*np.cos(2*np.pi * quefrency / T0)

tmp = np.zeros(fft_size, dtype=np.complex128)
tmp[:len(signal_spectrum)] = sinc_lifter * cos_lifter * signal_cepstrum
tmp = np.fft.fftshift(tmp)
estimation = np.exp(np.fft.fft(tmp))

plt.plot(np.abs(estimation))
plt.show()