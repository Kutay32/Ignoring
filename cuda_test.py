#!/usr/bin/env python3
"""
CUDA availability and functionality test
"""

import sys
import os

def test_cuda_basic():
    """Test basic CUDA availability"""
    print("🔍 Testing CUDA availability...")
    print("=" * 50)

    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")

        if cuda_available:
            print("✅ CUDA is available on this system!")

            # Get CUDA version
            cuda_version = torch.version.cuda
            print(f"CUDA Version: {cuda_version}")

            # Get device count
            device_count = torch.cuda.device_count()
            print(f"CUDA Devices: {device_count}")

            # Get current device
            current_device = torch.cuda.current_device()
            print(f"Current Device: {current_device}")

            # Test each device
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"Device {i}: {device_name} ({device_memory:.1f}GB)")

            # Test tensor operations on GPU
            print("\n🧪 Testing GPU tensor operations...")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                x = torch.randn(1000, 1000).to(device)
                y = torch.randn(1000, 1000).to(device)
                z = torch.mm(x, y)
                print("✅ GPU matrix multiplication successful!")

                # Test memory usage
                memory_allocated = torch.cuda.memory_allocated() / (1024**2)
                memory_reserved = torch.cuda.memory_reserved() / (1024**2)
                print(f"Memory Allocated: {memory_allocated:.1f} MB")
                print(f"Memory Reserved: {memory_reserved:.1f} MB")
        else:
            print("❌ CUDA is NOT available on this system")
            print("💡 Using CPU for computations (will be slower)")

            # Test CPU operations
            print("\n🧪 Testing CPU tensor operations...")
            x = torch.randn(1000, 1000)
            y = torch.randn(1000, 1000)
            z = torch.mm(x, y)
            print("✅ CPU matrix multiplication successful!")

    except ImportError:
        print("❌ PyTorch not installed!")
        print("Install with: pip install torch torchvision torchaudio")
        return False
    except Exception as e:
        print(f"❌ Error testing CUDA: {e}")
        return False

    print("=" * 50)
    return cuda_available

def test_transformers_cuda():
    """Test if transformers library can use CUDA"""
    print("\n🔍 Testing transformers CUDA support...")
    print("=" * 50)

    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")

        # Check if CUDA is available in transformers context
        import torch
        if torch.cuda.is_available():
            print("✅ Transformers can use CUDA")

            # Test model loading (lightweight test)
            print("🧪 Testing model loading on CUDA...")
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM

                # Use a very small model for testing
                model_name = "microsoft/DialoGPT-small"

                print(f"Loading {model_name} on CUDA...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

                # Quick inference test
                inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_length=10, pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                print("✅ Transformers CUDA test successful!")
                print(f"Sample output: {response}")

                # Clean up
                del model, tokenizer
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️  Model loading test failed: {e}")
                print("   This doesn't mean CUDA isn't working, just that the test model failed")
        else:
            print("❌ CUDA not available for transformers")

    except ImportError:
        print("❌ Transformers not installed!")
        print("Install with: pip install transformers accelerate")
        return False
    except Exception as e:
        print(f"❌ Error testing transformers CUDA: {e}")
        return False

    print("=" * 50)
    return True

def test_ultralytics_cuda():
    """Test if ultralytics (YOLO) can use CUDA"""
    print("\n🔍 Testing ultralytics CUDA support...")
    print("=" * 50)

    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")

        # Check CUDA in ultralytics context
        import torch
        if torch.cuda.is_available():
            print("✅ Ultralytics can potentially use CUDA")

            # Test YOLO model loading
            print("🧪 Testing YOLO model loading...")
            try:
                from ultralytics import YOLO

                # Test with your YOLO model
                model_path = "weights/yolov8s.pt"
                if os.path.exists(model_path):
                    print(f"Loading {model_path}...")
                    model = YOLO(model_path)

                    # Test inference on a simple image (create a dummy image)
                    import numpy as np
                    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

                    print("Running inference...")
                    results = model(dummy_image, verbose=False)

                    if results:
                        print("✅ YOLO CUDA test successful!")
                        print(f"Detections found: {len(results[0].boxes) if results[0].boxes is not None else 0}")
                    else:
                        print("⚠️  YOLO inference completed but no results")

                    # Clean up
                    del model
                    torch.cuda.empty_cache()

                else:
                    print(f"⚠️  YOLO weights not found at {model_path}")

            except Exception as e:
                print(f"⚠️  YOLO test failed: {e}")
                print("   This doesn't mean CUDA isn't working, just that YOLO test failed")
        else:
            print("❌ CUDA not available for ultralytics")

    except ImportError:
        print("❌ Ultralytics not installed!")
        print("Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Error testing ultralytics CUDA: {e}")
        return False

    print("=" * 50)
    return True

def main():
    """Main test function"""
    print("🖥️  CUDA System Test Suite")
    print("Testing CUDA availability and functionality")
    print("=" * 60)

    # Test basic CUDA
    cuda_available = test_cuda_basic()

    # Test transformers CUDA
    test_transformers_cuda()

    # Test ultralytics CUDA
    test_ultralytics_cuda()

    # Summary
    print("\n📊 SUMMARY")
    print("=" * 60)
    if cuda_available:
        print("✅ CUDA is AVAILABLE and WORKING!")
        print("🎯 Your system can run GPU-accelerated models")
        print("🚀 YOLOv8 + Qwen should work with GPU acceleration")
    else:
        print("❌ CUDA is NOT available")
        print("🐌 Models will run on CPU (slower but still works)")
        print("💡 Consider using CPU-only launcher: python cpu_ui_launcher.py")

    print("\n🔧 Recommendations:")
    if cuda_available:
        print("• Use: python stable_ui_launcher.py (GPU accelerated)")
        print("• Memory: ~8GB GPU RAM needed for Qwen2.5-VL-7B")
    else:
        print("• Use: python cpu_ui_launcher.py (CPU only)")
        print("• Memory: ~16GB+ RAM needed for Qwen2.5-VL-7B")

    print("=" * 60)

if __name__ == "__main__":
    main()
