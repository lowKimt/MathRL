# src/utils.py
import torch
import sys

# Attempt to import torch_directml if it's a separate package in your environment
# This is for older setups. Newer PyTorch might integrate DirectML differently.
try:
    import torch_directml
except ImportError:
    torch_directml = None # type: ignore

def get_device():
    """Gets the best available device, prioritizing DirectML, then CUDA, then MPS, then CPU."""
    if torch_directml and hasattr(torch_directml, 'device') and torch_directml.is_available():
        try:
            # Attempt to get the device name to confirm it works
            # DirectML device index might not be explicitly needed if only one exists.
            device_name_str = torch_directml.device_name(0) 
            print(f"Using DirectML device (via torch_directml): {device_name_str}")
            return torch_directml.device()
        except Exception as e:
            print(f"Could not initialize torch_directml device: {e}. Falling back.")
    
    # Check for DirectML support integrated into PyTorch (common in newer versions)
    # The attribute might be torch.dml or similar, check PyTorch DirectML plugin docs for your version
    if hasattr(torch, 'directml') and torch.directml.is_available():
        try:
            # device_id = torch.directml.current_device() # or simply use the string
            dml_device = torch.device("privateuseone:0") # Standard way to reference the first DirectML device
            # Test the device
            _ = torch.tensor([1.0]).to(dml_device)
            print(f"Using DirectML device (via torch.directml): {torch.directml.device_name(0)}")
            return dml_device
        except Exception as e:
            print(f"Could not initialize torch.directml device: {e}. Falling back.")
            
    if torch.cuda.is_available():
        print("Using CUDA device.")
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using MPS device (Apple Silicon).")
        return torch.device("mps")
    
    print("Using CPU.")
    return torch.device("cpu")

if __name__ == '__main__':
    # Test the device function
    device = get_device()
    print(f"Selected device: {device}")
    # Simple tensor operation to confirm device compatibility
    try:
        a = torch.tensor([1.0, 2.0]).to(device)
        b = torch.tensor([3.0, 4.0]).to(device)
        c = a + b
        print(f"Test operation on device {device} successful: {c.cpu().numpy()}")
    except Exception as e:
        print(f"Error during test operation on device {device}: {e}")