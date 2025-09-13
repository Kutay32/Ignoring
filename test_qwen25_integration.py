#!/usr/bin/env python3
"""
Test script to verify Qwen2.5-VL-7B integration
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from advanced_signature_detector import AdvancedSignatureDetector

def test_qwen25_integration():
    """Test Qwen2.5-VL-7B model loading and basic functionality"""
    print("Testing Qwen2.5-VL-7B integration...")

    # Initialize detector with Qwen2.5-VL-7B
    detector = AdvancedSignatureDetector(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        yolo_weights_path="weights/yolov8s.pt",
        use_quantization=True  # Enable quantization for memory efficiency
    )

    # Check if YOLO model loaded
    if detector.yolo_model is not None:
        print("✓ YOLOv8 model loaded successfully!")
    else:
        print("✗ YOLOv8 model failed to load")
        return False

    # Test Qwen model loading
    try:
        print("Testing Qwen2.5-VL-7B model loading...")
        detector.load_model()
        print("✓ Qwen2.5-VL-7B model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error loading Qwen2.5-VL-7B model: {e}")
        return False

if __name__ == "__main__":
    success = test_qwen25_integration()
    if success:
        print("\n🎉 Qwen2.5-VL-7B integration test passed!")
    else:
        print("\n❌ Qwen2.5-VL-7B integration test failed.")
    sys.exit(0 if success else 1)
