# inspect_checkpoint.py
#
# A simple script to load and inspect a PyTorch checkpoint.
# Edit the `checkpoint_path` variable below to point to your .pth.tar file.

import torch

# === User-defined path to your checkpoint ===
checkpoint_path = "wasb_basketball_best.pth.tar"  # <- replace with your actual path


def inspect_checkpoint(path):
    # Load the checkpoint on CPU
    ckpt = torch.load(path, map_location="cpu")
    print(f"Loaded checkpoint from: {path}")

    # Show top-level keys
    keys = list(ckpt.keys())
    print("Top-level keys in checkpoint:", keys)

    # Determine where the actual state_dict lives
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Print parameter names and shapes
    print("\nParameters in checkpoint:")
    total_params = 0
    for name, param in state_dict.items():
        if hasattr(param, 'shape'):
            shape = tuple(param.shape)
            print(f"  {name:60s} -> {shape}")
            total_params += param.numel()
        else:
            print(f"  {name:60s} -> (skipped, not a tensor)")

    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    inspect_checkpoint(checkpoint_path)

