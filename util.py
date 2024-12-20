import torch
import platform

# tries to use mps gpu, otherwise defaults to 
def get_device():
    os_name = platform.system()
    
    # First check for CUDA availability as it's generally preferred when available
    if torch.cuda.is_available():
        print("CUDA available...")
        return torch.device("cuda")
    
    # on mac try to use mps
    if os_name == 'Darwin':
        if torch.backends.mps.is_available():
            print("MPS available...")
            return torch.device("mps")
        else:
            if torch.backends.mps.is_built():
                print("MPS not available because the current MacOS version is not 12.3+ "
                      "and/or you do not have an MPS-enabled device on this machine.")
            else:
                print("MPS not available because the current PyTorch install was not "
                      "built with MPS enabled.")
            print("Using cpu device for mac os...")
            return torch.device("cpu")
            
    # on windows try to use DirectML
    if os_name == 'Windows':
        import torch_directml
        try:
            dml = torch_directml.device()
            print("DirectML available...")
            return dml
        except Exception as e:
            print(f"DirectML is not available: {e}")
            print("Using cpu device for windows os...")
            return torch.device("cpu")
    
    print("Using cpu device...")
    return torch.device("cpu")