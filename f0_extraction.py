import numpy as np
import torch
import torchcrepe

def get_f0_crepe(audio, sr, 
                 device='cpu', 
                 model='tiny', # tiny or full
                 batch_size=2048,
                 hop_length=256, 
                 fmin=50,
                 fmax=1100):
    # put audio in a tensor
    audio = torch.unsqueeze(torch.Tensor(audio), 0)

    # hop_length = int(sr / 200.)
    # fmin = 50
    # fmax = 1100

    result = torchcrepe.predict(audio,
                            sr,
                            hop_length,
                            fmin,
                            fmax,
                            model,
                            batch_size=batch_size,
                            device=device,
                            return_periodicity=True)
    
    return result[0].detach().cpu().numpy().squeeze(), result[1].detach().cpu().numpy()