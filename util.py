import torch

# tries to use mps gpu, otherwise defaults to 
def get_device(): 
    device = torch.device("cpu")
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
    else:
        print("MPS available...")
        device = torch.device("mps")
    return device
