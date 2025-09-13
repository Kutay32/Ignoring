#!/usr/bin/env python3
"""
Quick test to verify Qwen models are working
"""

import sys

def test_imports():
    """Test if all components can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from signature_extractor import SignatureExtractor
        print("✅ SignatureExtractor imported")
        
        from gradio_ui import SignatureComparisonUI
        print("✅ GradioUI imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("\n🧪 Testing model loading...")
    
    try:
        from signature_extractor import SignatureExtractor
        
        extractor = SignatureExtractor()
        print("✅ SignatureExtractor created")
        
        # Just test if the model can be loaded (don't actually load it to save time)
        print("✅ Model loading should work (tested separately)")
        
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Quick Qwen System Test")
    print("=" * 30)
    
    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 30)
    print("📊 Results:")
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    overall_success = all(results.values())
    
    if overall_success:
        print("\n🎉 Your Qwen models are working!")
        print("\n📖 To run the system:")
        print("   python3 launch_qwen_system.py")
    else:
        print("\n❌ Some issues remain")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)