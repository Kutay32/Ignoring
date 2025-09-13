#!/usr/bin/env python3
"""
Test script to verify the Qwen model fix
"""

import torch
import sys
from signature_extractor import SignatureExtractor

def test_signature_extractor():
    """Test the fixed signature extractor"""
    print("🧪 Testing Fixed Signature Extractor...")
    
    try:
        # Test with 2B model (most likely to work)
        print("Testing with Qwen2-VL-2B-Instruct...")
        extractor = SignatureExtractor("Qwen/Qwen2-VL-2B-Instruct")
        
        # Test model loading
        print("Loading model...")
        extractor.load_model()
        print("✅ Model loaded successfully!")
        
        # Test that the model has the generate method
        if hasattr(extractor.model, 'generate'):
            print("✅ Model has generate method!")
        else:
            print("❌ Model does not have generate method!")
            return False
        
        # Test basic generation (without image for now)
        print("Testing basic text generation...")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, can you see this text?"}
                ]
            }
        ]
        
        text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = extractor.processor(
            text=[text],
            images=None,
            videos=None,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            generated_ids = extractor.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = extractor.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"✅ Generation successful: {response}")
        
        print("🎉 All tests passed! The fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Testing Fixed Qwen Model")
    print("=" * 50)
    
    success = test_signature_extractor()
    
    if success:
        print("\n✅ SUCCESS: The Qwen model fix is working correctly!")
        print("The 'generate' method is now available and functional.")
    else:
        print("\n❌ FAILED: There are still issues with the Qwen model.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)