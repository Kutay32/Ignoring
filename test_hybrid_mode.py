#!/usr/bin/env python3
"""
Test the new Hybrid Mode: YOLO detection + targeted Qwen analysis
"""

import os
import time
from modern_ui_backend import ModernSignatureAPI

def test_hybrid_mode():
    """Test the hybrid YOLO + Qwen processing mode"""
    print("🧪 Testing Hybrid Mode (YOLO + Qwen)")
    print("=" * 50)

    # Initialize API
    api = ModernSignatureAPI()

    # Test with a sample image if available
    test_images = []
    if os.path.exists("exported"):
        test_images = [os.path.join("exported", f) for f in os.listdir("exported") if f.endswith('.jpg')][:1]
    elif os.path.exists("Ignoring/exported"):
        test_images = [os.path.join("Ignoring/exported", f) for f in os.listdir("Ignoring/exported") if f.endswith('.jpg')][:1]

    if not test_images:
        print("❌ No test images found in Ignoring/exported/")
        print("Please add some signature images to test the hybrid mode.")
        return

    image_path = test_images[0]
    print(f"📸 Testing with: {os.path.basename(image_path)}")

    # Load model first
    print("🤖 Loading Qwen model...")
    api.detector.load_model()

    # Test hybrid processing
    print("🎯 Running Hybrid Mode processing...")
    start_time = time.time()

    try:
        # Simulate the hybrid mode processing
        import cv2
        from PIL import Image

        # Read image
        image = cv2.imread(image_path)
        pil_image = Image.open(image_path)

        # Save temp file
        temp_path = f"/tmp/test_hybrid_{int(time.time())}.png"
        pil_image.save(temp_path)

        # Step 1: YOLO detection
        print("🔍 Step 1: YOLO signature detection...")
        regions = api.detector.detect_signature_regions_yolo(temp_path)
        print(f"   Found {len(regions)} potential signature regions")

        # Step 2: Targeted Qwen analysis on each region
        print("🧠 Step 2: Targeted Qwen analysis on each cropped signature...")
        signatures = []
        for i, region in enumerate(regions[:2]):  # Test with first 2 regions
            print(f"   Analyzing signature {i+1}/{min(2, len(regions))}")
            x1, y1, x2, y2 = region['bbox']
            cropped = pil_image.crop((x1, y1, x2, y2))

            # Quick Qwen analysis
            analysis_result = api._quick_signature_analysis(cropped, region)

            signatures.append({
                "id": i + 1,
                "type": analysis_result.get("signature_type", "detected"),
                "confidence": region['confidence'],
                "qwen_analysis": analysis_result.get("analysis", "Analysis completed"),
                "quality_score": analysis_result.get("quality_score", 0.5)
            })

        processing_time = time.time() - start_time

        # Results
        print("\n✅ Hybrid Mode Test Results:")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"   Regions detected: {len(regions)}")
        print(f"   Signatures analyzed: {len(signatures)}")

        for sig in signatures:
            print(f"   Signature {sig['id']}: {sig['type']} (confidence: {sig['confidence']:.2f}, quality: {sig['quality_score']:.2f})")
            print(f"      Qwen: {sig['qwen_analysis'][:100]}...")

        # Performance check
        if processing_time < 120:  # Less than 2 minutes
            print("🎉 SUCCESS: Hybrid mode is much faster than 5-hour processing!")
        else:
            print("⚠️  WARNING: Processing still took longer than expected")

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hybrid_mode()
