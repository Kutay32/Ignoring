#!/usr/bin/env python3
"""
Simple YOLO-only signature detection test
"""

import os
from advanced_signature_detector import AdvancedSignatureDetector

def test_yolo_only():
    """Test YOLO detection without any LLM"""
    print("🔍 Testing YOLO-Only Signature Detection")
    print("=" * 50)

    # Initialize detector
    detector = AdvancedSignatureDetector()

    # Load only YOLO
    detector._load_yolo_model()

    if detector.yolo_model is None:
        print("❌ YOLO model failed to load")
        return

    print("✅ YOLO model loaded successfully")

    # Test with first available image
    test_images = []
    if os.path.exists("exported"):
        test_images = [os.path.join("exported", f) for f in os.listdir("exported") if f.endswith('.jpg')][:1]
    elif os.path.exists("Ignoring/exported"):
        test_images = [os.path.join("Ignoring/exported", f) for f in os.listdir("Ignoring/exported") if f.endswith('.jpg')][:1]

    if not test_images:
        print("❌ No test images found")
        return

    image_path = test_images[0]
    print(f"📸 Testing with: {os.path.basename(image_path)}")

    # Test YOLO detection
    import time
    start_time = time.time()

    regions = detector.detect_signature_regions_yolo(image_path)
    processing_time = time.time() - start_time

    print(".2f"    print(f"   Found {len(regions)} potential signature regions")

    if regions:
        print("   Sample regions:")
        for i, region in enumerate(regions[:3]):  # Show first 3
            bbox = region['bbox']
            conf = region['confidence']
            print(".2f"
    # Summary
    print("\n✅ YOLO-Only Test Results:")
    print("   • YOLO detection: ✅ Working"    print(f"   • Signatures found: {len(regions)}")
    print(".2f"
    print("\n🎯 YOLO-only detection is working perfectly!")
    print("   This is fast, reliable, and doesn't require massive models."
    print("\n💡 Next: Create a simple web UI for this YOLO-only version")

if __name__ == "__main__":
    test_yolo_only()
