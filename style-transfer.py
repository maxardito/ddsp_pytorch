import torch
import numpy as np
from scipy.io import wavfile
import numpy as np

# Parameters for pitch glissando
duration = 3.0  # Duration in seconds
sampling_rate = 44100  # Sampling rate (samples per second)
num_samples = int(duration * sampling_rate)
time = np.linspace(0, duration, num_samples)

# Parameters for loudness contour
attack_time = 0.2  # Attack time in seconds
decay_time = 0.5  # Decay time in seconds

# Generate pitch glissando
start_pitch = 200  # Starting pitch in Hz
end_pitch = 400  # Ending pitch in Hz
pitch = np.exp(np.linspace(np.log(start_pitch), np.log(end_pitch),
                           num_samples))

# Generate loudness contour
loudness = np.ones_like(time)
attack_samples = int(attack_time * sampling_rate)
decay_samples = int(decay_time * sampling_rate)

# Apply exponential attack and decay to loudness
attack_curve = np.linspace(0, 1, attack_samples)
decay_curve = np.linspace(1, 0, decay_samples)
loudness[:attack_samples] *= attack_curve
loudness[-decay_samples:] *= decay_curve

# Normalize loudness between 0 and 1
loudness = (loudness - np.min(loudness)) / (np.max(loudness) -
                                            np.min(loudness))

pitch = torch.unsqueeze(torch.tensor(pitch), dim=1)
loudness = torch.unsqueeze(torch.tensor(loudness), dim=1)

loudness = torch.unsqueeze(loudness, dim=0)

model = torch.jit.load("./grimes/ddsp_grimes_pretrained.ts")

audio = model(pitch.to(torch.float), loudness.to(torch.float))

# Specify the sample rate and the file path
sample_rate = 44100
file_path = 'output.wav'

# Scale the audio data to the appropriate range for the desired data type (e.g., int16)
scaled_data = np.int16(audio.detach().numpy() *
                       32767)  # Scale to the range of 16-bit signed integers

# Write the WAV file
wavfile.write(file_path, sample_rate, scaled_data)
