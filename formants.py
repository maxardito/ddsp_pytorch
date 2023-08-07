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
    

sample_rate = 16000
dur = 1
num_samples = int(dur * sample_rate)
t = np.linspace(0, dur, num_samples, False)

# source
source = np.zeros(num_samples)
f0 = 130
fk = f0
while fk < sample_rate/2:
    source += np.cos(2.0*np.pi*fk * t)
    fk += f0

source *= sig.windows.hann(num_samples)

# filter
fft_size = (num_samples*2)-1
f = np.linspace(-sample_rate/2, sample_rate/2, fft_size, False)
formants = [400, 800]
filter_spectrum = np.zeros(fft_size)
for formant in formants:
    filter_spectrum += gaussian(f, formant, 100)
source_spectrum = np.fft.fft(source, fft_size)
source_spectrum = np.fft.fftshift(source_spectrum)
mixture_spectrum = filter_spectrum * source_spectrum
mixture_spectrum = np.fft.fftshift(mixture_spectrum)
mixture = np.real(np.fft.ifft(mixture_spectrum))

mixture = normalize(mixture)
sf.write("tmp.wav", mixture, samplerate=sample_rate)

plt.subplot(2, 1, 1)
plt.plot(mixture)
plt.subplot(2, 1, 2)
plt.plot(f, 20*np.log10(np.abs(source_spectrum)))
plt.ylim([-100, 30])
plt.show()