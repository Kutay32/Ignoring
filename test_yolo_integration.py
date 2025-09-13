#!/usr/bin/env python3
"""
Test script to verify YOLOv8 integration with signature detection
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from advanced_signature_detector import AdvancedSignatureDetector
import cv2
import numpy as np

def test_yolo_integration():
    """Test YOLO model loading and basic detection"""
    print("Testing YOLOv8 integration with signature detection...")

    # Initialize detector
    detector = AdvancedSignatureDetector()

    # Check if YOLO model loaded
    if detector.yolo_model is not None:
        print("✓ YOLOv8 model loaded successfully!")
    else:
        print("✗ YOLOv8 model failed to load")
        return False

    # Create a simple test image with some shapes that might be detected
    test_image = np.ones((800, 1200, 3), dtype=np.uint8) * 255  # White background

    # Add some rectangles that could represent signatures
    cv2.rectangle(test_image, (100, 100), (300, 150), (0, 0, 0), -1)  # Black rectangle
    cv2.rectangle(test_image, (500, 200), (700, 250), (0, 0, 0), -1)  # Another black rectangle

    # Add some text-like shapes
    cv2.putText(test_image, "SIGNATURE", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Save test image
    test_image_path = "test_signature_image.jpg"
    cv2.imwrite(test_image_path, test_image)
    print(f"Created test image: {test_image_path}")

    # Test detection
    try:
        regions = detector.detect_signature_regions_yolo(test_image_path)
        print(f"✓ Detection completed! Found {len(regions)} potential signature regions")

        for i, region in enumerate(regions):
            print(f"  Region {i+1}: bbox={region['bbox']}, confidence={region['confidence']:.3f}, method={region['method']}")

        # Test Qwen model loading
        print("Testing Qwen model loading...")
        detector.load_model()
        print("✓ Qwen model loaded successfully!")

        # Test feature extraction on first region if any found
        if regions:
            print("Testing signature feature extraction with Qwen...")
            features = detector.extract_signature_features_advanced(test_image_path, regions[0]['bbox'])
            print("✓ Feature extraction completed!")
            print(f"  Signature type: {features['features']['signature_type']}")
            print(f"  Confidence score: {features['confidence_score']:.3f}")

        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

        return True

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        return False

if __name__ == "__main__":
    success = test_yolo_integration()
    if success:
        print("\n🎉 All tests passed! YOLO + Qwen integration is working.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    sys.exit(0 if success else 1)
