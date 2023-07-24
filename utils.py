import numpy as np

def normalize(signal):
    if np.max(np.abs(signal)) is not 0.0:
        return signal / np.max(np.abs(signal))
    else:
        return signal

def generate_glissando():
    # Parameters for pitch glissando
    duration = 0.5  # Duration in seconds
    sampling_rate = 44100  # Sampling rate (samples per second)
    num_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, num_samples)

    # Parameters for loudness contour
    attack_time = 0.5  # Attack time in seconds
    decay_time = 0.05  # Decay time in seconds

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
    
    return pitch, loudness