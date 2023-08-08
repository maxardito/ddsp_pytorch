import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import soundfile as sf
import opensmile
import pandas as pd

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

sample_rate = 32000
dur = 2
num_samples = int(dur * sample_rate)
t = np.linspace(0, dur, num_samples, False)

# source audio
f0 = 130
formants = [400, 8000, 1500]
signal = source_filter_model(t, f0=f0, formants=formants)

# analysis 
# smile = opensmile.Smile(
#     feature_set=opensmile.FeatureSet.ComParE_2016,
#     feature_level=opensmile.FeatureLevel.Functionals,
# )
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)
smile.process_signal(
    signal,
    sample_rate
)

data = smile.process_signal(signal, sample_rate)

centerformantfreqs = ['F1frequency_sma3nz', 'F2frequency_sma3nz', 'F3frequency_sma3nz']
formant_df = data[centerformantfreqs]

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(formant_df)


signal_spectrum = np.fft.fft(signal)
signal_spectrum = signal_spectrum[:len(signal_spectrum)//2]
signal_spectrum = 20*np.log10(np.abs(signal_spectrum))
plt.plot(signal_spectrum)
plt.show()