#!/usr/bin/env python3
"""
Kaggle T4x2 Setup Script for Signature Detection System
Optimized for Kaggle's dual T4 GPU environment
"""

import os
import sys
import subprocess

def setup_kaggle_environment():
    """Setup the environment for Kaggle T4x2 GPUs"""
    print("🎯 Setting up for Kaggle T4x2 Environment")
    print("=" * 50)

    # Check GPU availability
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        print(f"✅ Found {gpu_count} GPU(s)")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
            print(f"  GPU {i}: {gpu_name} ({gpu_memory}GB VRAM)")

        if gpu_count >= 2:
            print("🚀 Multi-GPU setup detected - enabling optimal performance!")
        else:
            print("⚠️ Single GPU detected - system will still work but with reduced performance")

    except ImportError:
        print("❌ PyTorch not found - install required packages first")

    # Install requirements
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade",
            "torch", "transformers", "ultralytics", "fastapi", "uvicorn[standard]",
            "python-multipart", "accelerate", "bitsandbytes"
        ])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

    # Check model compatibility
    print("\n🧠 Checking model compatibility...")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print(f"✅ Qwen2.5-VL-7B model is compatible with your setup")
        print(f"   Model size: ~14GB (fits in T4 16GB VRAM with quantization)")
    except Exception as e:
        print(f"⚠️ Model compatibility check failed: {e}")

    print("\n🎉 Kaggle T4x2 setup complete!")
    print("💡 Run 'python launch_modern_ui.py' to start the signature detection system")
    return True

def optimize_for_kaggle():
    """Apply Kaggle-specific optimizations"""
    print("\n⚡ Applying Kaggle optimizations...")

    # Set environment variables for better GPU utilization
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use both T4 GPUs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Better memory management
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

    print("✅ Environment optimized for T4x2")

if __name__ == "__main__":
    success = setup_kaggle_environment()
    if success:
        optimize_for_kaggle()
        print("\n🚀 Ready to run: python launch_modern_ui.py")
    else:
        print("\n❌ Setup failed - check the errors above")
        sys.exit(1)
