#!/usr/bin/env python3
"""
Test the fixed Qwen model loading
"""

import sys
from signature_extractor import SignatureExtractor

def test_signature_extractor():
    """Test the SignatureExtractor with the fixed configuration"""
    print("🧪 Testing Fixed SignatureExtractor...")
    
    try:
        # Test with default model (2B)
        print("Testing with default 2B model...")
        extractor = SignatureExtractor()
        extractor.load_model()
        print("✅ 2B model loaded successfully!")
        
        # Test with 7B model (should fallback to 2B)
        print("\nTesting with 7B model (should fallback to 2B)...")
        extractor_7b = SignatureExtractor("Qwen/Qwen2.5-VL-7B-Instruct")
        extractor_7b.load_model()
        print("✅ 7B model fallback successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_interface():
    """Test if the web interface can be imported and initialized"""
    print("\n🧪 Testing Web Interface...")
    
    try:
        from gradio_ui import SignatureComparisonUI
        ui = SignatureComparisonUI()
        print("✅ Web interface initialized successfully!")
        print(f"Available models: {list(ui.models.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Fixed Qwen Models")
    print("=" * 50)
    
    success1 = test_signature_extractor()
    success2 = test_web_interface()
    
    if success1 and success2:
        print("\n🎉 All tests PASSED! Your Qwen models should work now.")
        print("\n📖 To run the system:")
        print("   python3 run_system.py --mode ui")
        return True
    else:
        print("\n❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)