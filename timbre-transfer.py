import torch
import numpy as np
from scipy.io import wavfile
from effortless_config import Config
from ddsp.core import extract_centroid, extract_loudness, extract_pitch
import numpy as np
import yaml
import librosa


def extract_features(
    signal,
    sampling_rate,
    block_size,
):

    p = extract_pitch(signal, sampling_rate, block_size)
    # c = extract_centroid(signal, sampling_rate, block_size)
    l = extract_loudness(signal, block_size)

    # return p, c, l
    return p, l


def main():

    class args(Config):
        CONFIG = "config.yaml"

    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    # Load the model
    model = torch.jit.load(config["timbre_transfer"]["model"])

    # Load the audio file
    signal, sampling_rate = librosa.load(
        config["timbre_transfer"]["input_file"])

    # p, c, l = extract_features(signal, sampling_rate,
    p, l = extract_features(signal, sampling_rate,
                            config["model"]["block_size"])

    p = torch.tensor(p, dtype=torch.float).unsqueeze(0).unsqueeze(2)
    # c = torch.tensor(c, dtype=torch.float).unsqueeze(0).unsqueeze(2)
    l = torch.tensor(l, dtype=torch.float).unsqueeze(0).unsqueeze(2)

    # y = model(p, c, l)
    y = model(p, l)

    audio = y.squeeze().detach().numpy()

    # Scale the audio data to the appropriate range for the desired data type (e.g., int16)
    scaled_data = np.int16(
        audio * 32767)  # Scale to the range of 16-bit signed integers

    # Write the WAV file
    file_path = 'output.wav'
    wavfile.write(file_path, sampling_rate, scaled_data)


if __name__ == "__main__":
    main()
