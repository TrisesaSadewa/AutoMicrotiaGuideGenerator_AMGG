import sys
import os

print("🔄 Forcing re-installation of compatible PyTorch versions...")

# 1. Uninstall everything related to torch first to clean up potential mess
!{sys.executable} -m pip uninstall -y torch torchvision torchaudio

# 2. Install the specific compatible pair (Torch 2.0.1 + Torchvision 0.15.2 for CUDA 11.7)
# We use --no-cache-dir to avoid picking up broken cached wheels
!{sys.executable} -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117 --no-cache-dir

print("\n✅ Installation complete. PLEASE RESTART THE KERNEL NOW.")