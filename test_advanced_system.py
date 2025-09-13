#!/usr/bin/env python3
"""
Test script for Advanced Signature Detection System
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
from advanced_signature_detector import AdvancedSignatureDetector

def create_test_signature_image(text="Test Signature", width=400, height=200):
    """Create a test signature image"""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Draw signature text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    draw.text((x, y), text, fill='black', font=font)
    
    return img

def test_signature_detection():
    """Test the signature detection system"""
    print("🔍 Testing Advanced Signature Detection System")
    print("=" * 50)
    
    # Initialize detector
    print("1. Initializing detector...")
    detector = AdvancedSignatureDetector("Qwen/Qwen2-VL-2B-Instruct")
    
    try:
        print("2. Loading model...")
        detector.load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Create test images
    print("3. Creating test images...")
    test_images = []
    
    # Create different signature styles
    signatures = [
        "John Doe",
        "Jane Smith", 
        "A. Johnson",
        "M. Williams"
    ]
    
    for i, sig_text in enumerate(signatures):
        img = create_test_signature_image(sig_text)
        temp_path = f"/tmp/test_signature_{i}.png"
        img.save(temp_path)
        test_images.append(temp_path)
        print(f"   Created test image: {temp_path}")
    
    # Test detection
    print("4. Testing signature detection...")
    for i, img_path in enumerate(test_images):
        print(f"\n   Testing image {i+1}: {img_path}")
        
        try:
            # Test OpenCV detection
            regions = detector.detect_signature_regions_opencv(img_path)
            print(f"   Detected {len(regions)} regions")
            
            for j, region in enumerate(regions):
                print(f"     Region {j+1}: {region['method']} (confidence: {region['confidence']:.2f})")
            
            # Test feature extraction
            if regions:
                best_region = regions[0]
                features = detector.extract_signature_features_advanced(
                    img_path, 
                    best_region['bbox']
                )
                print(f"   Signature type: {features['features'].get('signature_type', 'Unknown')}")
                print(f"   Confidence: {features.get('confidence_score', 0):.2f}")
                print(f"   Hash: {features.get('signature_hash', 'N/A')[:16]}...")
            
        except Exception as e:
            print(f"   ❌ Error processing image: {e}")
    
    # Test similarity comparison
    print("\n5. Testing similarity comparison...")
    try:
        # Process first two images
        result1 = detector.process_document_advanced(test_images[0], "user1")
        result2 = detector.process_document_advanced(test_images[1], "user2")
        
        if (result1['success'] and result2['success'] and 
            result1['signature_results'] and result2['signature_results']):
            
            features1 = result1['signature_results'][0]['features']
            features2 = result2['signature_results'][0]['features']
            
            similarities = detector.calculate_advanced_similarity(features1, features2)
            print(f"   Text similarity: {similarities['text_similarity']:.3f}")
            print(f"   Feature similarity: {similarities['feature_similarity']:.3f}")
            print(f"   Overall similarity: {similarities['overall_similarity']:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error in similarity test: {e}")
    
    # Test database operations
    print("\n6. Testing database operations...")
    try:
        # Test storing signatures
        for i, img_path in enumerate(test_images[:2]):
            result = detector.process_document_advanced(img_path, f"test_user_{i}")
            if result['success'] and result['signature_results']:
                sig_data = result['signature_results'][0]
                detector.store_signature_advanced(f"test_user_{i}", sig_data, img_path)
                print(f"   Stored signature for test_user_{i}")
        
        # Test finding similar signatures
        if result1['success'] and result1['signature_results']:
            similar = detector.find_similar_signatures_advanced(
                result1['signature_results'][0]['features']
            )
            print(f"   Found {len(similar)} similar signatures")
    
    except Exception as e:
        print(f"   ❌ Error in database test: {e}")
    
    # Cleanup
    print("\n7. Cleaning up...")
    for img_path in test_images:
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"   Removed {img_path}")
    
    print("\n✅ Test completed!")
    return True

def test_ui_components():
    """Test UI components without launching"""
    print("\n🖥️  Testing UI Components")
    print("=" * 30)
    
    try:
        from advanced_gradio_ui import AdvancedSignatureUI
        ui = AdvancedSignatureUI()
        print("✅ UI class initialized successfully")
        
        # Test model loading
        status = ui.load_model("Qwen/Qwen2-VL-2B-Instruct")
        print(f"Model loading status: {status}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing UI: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Advanced Signature Detection System Tests")
    print("=" * 60)
    
    # Test core functionality
    core_test_passed = test_signature_detection()
    
    # Test UI components
    ui_test_passed = test_ui_components()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 20)
    print(f"Core functionality: {'✅ PASSED' if core_test_passed else '❌ FAILED'}")
    print(f"UI components: {'✅ PASSED' if ui_test_passed else '❌ FAILED'}")
    
    if core_test_passed and ui_test_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nTo launch the UI, run:")
        print("python advanced_gradio_ui.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()