#!/usr/bin/env python3
"""
Test script to verify the smaller Qwen model loads correctly
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def test_model_loading():
    """Test loading the smaller Qwen model"""
    print("🧪 Testing Qwen2.5-VL-3B-Instruct model loading...")

    # Check hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📊 Using device: {device}")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        print(f"🎮 GPU: {gpu_name} ({gpu_memory}GB VRAM)")

    try:
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"📥 Loading {model_name}...")
        print("⏳ This may take a few minutes...")

        # Load processor
        print("🔧 Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Load model with quantization if CUDA available
        if device == "cuda":
            try:
                print("⚡ Attempting 8-bit quantization...")
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("✅ Model loaded with 8-bit quantization!")
            except Exception as quant_error:
                print(f"⚠️ Quantization failed: {quant_error}")
                print("🔄 Falling back to standard loading...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("✅ Model loaded with standard approach!")
        else:
            print("💻 Loading on CPU...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(device)
            print("✅ Model loaded on CPU!")

        print("🎉 Model loading test PASSED!")
        print("💡 The smaller model should work with your RTX 3080 Ti (11GB VRAM)")

        # Clean up
        del model
        del processor
        if device == "cuda":
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Model loading test FAILED: {e}")
        print("🔧 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Try the even smaller Qwen2-VL-2B-Instruct model")
        print("3. Consider using CPU-only mode")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    exit(0 if success else 1)
