#!/usr/bin/env python3
"""
Test script to check if Qwen models can be loaded
"""

import torch
import sys
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def test_qwen_loading():
    """Test loading Qwen 2.5-VL model"""
    print("🧪 Testing Qwen 2.5-VL Model Loading...")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Test model loading
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"\n📦 Loading model: {model_name}")
    
    try:
        # Load processor
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Processor loaded successfully")
        
        # Load model
        print("Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")
        
        # Test basic functionality
        print("\n🔍 Testing basic functionality...")
        
        # Create a simple test message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, can you see this text?"}
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"✅ Chat template applied: {text[:100]}...")
        
        # Tokenize
        image_inputs, video_inputs = processor(
            text=[text],
            images=None,
            videos=None,
            return_tensors="pt"
        )
        print("✅ Text tokenized successfully")
        
        # Test generation (short)
        print("Testing generation...")
        with torch.no_grad():
            generated_ids = model.generate(
                **image_inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(image_inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"✅ Generation successful: {response}")
        
        print("\n🎉 Qwen model loading and basic functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading Qwen model: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test if all required imports work"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("✅ torch imported")
        
        import cv2
        print("✅ cv2 imported")
        
        import numpy as np
        print("✅ numpy imported")
        
        from PIL import Image
        print("✅ PIL imported")
        
        import json
        print("✅ json imported")
        
        import sqlite3
        print("✅ sqlite3 imported")
        
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        print("✅ transformers imported")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported")
        
        import seaborn as sns
        print("✅ seaborn imported")
        
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✅ scikit-learn imported")
        
        import gradio as gr
        print("✅ gradio imported")
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Qwen Model Test")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("❌ Import test failed - cannot proceed")
        return False
    
    print("\n" + "=" * 50)
    
    # Test model loading
    if not test_qwen_loading():
        print("❌ Qwen model test failed")
        return False
    
    print("\n" + "=" * 50)
    print("🎯 All tests PASSED! Qwen models should work now.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)