#!/usr/bin/env python3
"""
Simple test to verify Qwen model loading works
"""

import torch
from transformers import AutoModel, AutoProcessor

def test_simple_loading():
    """Test simple model loading"""
    print("Testing Qwen model loading...")
    
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    
    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Processor loaded")
        
        print("Loading model...")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True
        )
        print("✅ Model loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_loading()
    if success:
        print("🎉 Qwen models are working!")
    else:
        print("❌ Qwen models still have issues")