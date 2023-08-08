import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import soundfile as sf
import parselmouth
from parselmouth import praat
import librosa

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
formants = [400, 800, 1500]
signal = source_filter_model(t, f0=f0, formants=formants)

sf.write("tmp.wav", signal, sample_rate)

tmp = parselmouth.Sound("tmp.wav")

pointProcess = praat.call(tmp, "To PointProcess (periodic, cc)", 75, 400)
formants = praat.call(tmp, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)

numPoints = praat.call(pointProcess, "Get number of points")
f1_list = []
f2_list = []
f3_list = []
for point in range(0, numPoints):
    point += 1
    t = praat.call(pointProcess, "Get time from index", point)
    f1 = praat.call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
    f2 = praat.call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
    f3 = praat.call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
    f1_list.append(f1)
    f2_list.append(f2)
    f3_list.append(f3)

# signal_spectrum = np.fft.fft(signal)
# signal_spectrum = signal_spectrum[:len(signal_spectrum)//2]
# signal_spectrum = 20*np.log10(np.abs(signal_spectrum))
# plt.plot(signal_spectrum)
# plt.show()

# signal_spectrogram = np.abs(librosa.stft(signal))
# spect_t = np.linspace(0, dur, signal_spectrogram.shape[0])
# spect_f = np.linspace(0, sample_rate/2, signal_spectrogram.shape[1], False)
# signal_spectrogram = librosa.amplitude_to_db(signal_spectrogram, ref=np.max)
# plt.pcolormesh(spect_t, spect_f, signal_spectrogram, shading='gouraud')

f, t, Sxx = sig.spectrogram(signal, sample_rate)
plt.pcolormesh(t, f, Sxx, shading='gouraud')


plt.show()