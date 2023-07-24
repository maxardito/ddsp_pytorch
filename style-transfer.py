import torch
import numpy as np
from scipy.io import wavfile
import numpy as np
import soundfile as sf
import torchcrepe
from f0_extraction import get_f0_crepe
from loudness_extraction import get_loudness
from utils import normalize, vector2tensor

# filename = "path/to/file.wav"
filename = "/Users/julianvanasse/Music/test-audio/billions.wav"

# read file
audio, sample_rate = sf.read(filename)
# force to mono
audio = np.mean(audio, axis=1) if audio.shape[1] > 1 else audio
# get dims
num_samples = len(audio)
duration = num_samples / sample_rate
time = np.linspace(0, duration, num_samples, False)

# extract f0
pitch, _ = get_f0_crepe(audio, sample_rate)
num_frames = len(pitch)
hop_size = num_samples / num_frames

# extract loudness
loudness = get_loudness(audio, hop_size)
# normalize
loudness = normalize(loudness)

# convert to 
pitch = vector2tensor(pitch)
loudness = vector2tensor(loudness)

model = torch.jit.load("./export/grimes/ddsp_grimes_pretrained.ts")

audio = model(pitch, loudness)

# Specify the sample rate and the file path
sample_rate = 44100
file_path = 'output.wav'

# Scale the audio data to the appropriate range for the desired data type (e.g., int16)
scaled_data = np.int16(audio.squeeze().detach().numpy() *
                       32767)  # Scale to the range of 16-bit signed integers
# Write the WAV file
wavfile.write(file_path, sample_rate, scaled_data)
