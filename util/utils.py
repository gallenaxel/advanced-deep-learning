import torch

def get_device():
    # find device
    if torch.cuda.is_available(): # NVIDIA
        device = torch.device('cuda')
    elif torch.backends.mps.is_available(): # apple M1/M2
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device