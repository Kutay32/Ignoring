#!/usr/bin/env python3
"""
Test script for the Signature Extraction and Comparison System
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import json
from signature_extractor import SignatureExtractor

def create_test_signature_image(width=800, height=600, signature_text="John Doe"):
    """Create a test signature image for testing"""
    # Create a white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Add some document content
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add document header
    draw.text((50, 50), "OFFICIAL DOCUMENT", fill='black', font=font)
    draw.text((50, 80), "Date: 2024-01-15", fill='black', font=font)
    draw.text((50, 110), "Document ID: DOC-12345", fill='black', font=font)
    
    # Add some content
    draw.text((50, 150), "This is a test document for signature verification.", fill='black', font=font)
    draw.text((50, 180), "The signature below should be extracted and analyzed.", fill='black', font=font)
    
    # Add a simulated stamp (circular)
    stamp_center = (200, 300)
    stamp_radius = 40
    draw.ellipse([stamp_center[0]-stamp_radius, stamp_center[1]-stamp_radius,
                  stamp_center[0]+stamp_radius, stamp_center[1]+stamp_radius],
                 outline='red', width=3)
    draw.text((stamp_center[0]-30, stamp_center[1]-10), "APPROVED", fill='red', font=font)
    
    # Add signature area
    signature_y = 450
    draw.text((50, signature_y), "Signature:", fill='black', font=font)
    
    # Draw a simulated signature (curved lines)
    signature_x = 150
    signature_width = 200
    signature_height = 50
    
    # Create curved signature lines
    points = []
    for i in range(0, signature_width, 5):
        x = signature_x + i
        y = signature_y + 20 + int(10 * np.sin(i * 0.1)) + int(5 * np.cos(i * 0.2))
        points.append((x, y))
    
    if len(points) > 1:
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill='blue', width=2)
    
    # Add signature text
    draw.text((signature_x, signature_y + 30), signature_text, fill='blue', font=font)
    
    return img

def test_signature_extraction():
    """Test the signature extraction functionality"""
    print("🧪 Testing Signature Extraction System...")
    
    # Create test image
    print("📝 Creating test signature image...")
    test_img = create_test_signature_image()
    
    # Save test image
    test_path = "test_signature.png"
    test_img.save(test_path)
    print(f"✅ Test image saved as {test_path}")
    
    # Test with different models
    models_to_test = [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        # Note: 32B and 72B models require more resources
    ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n🔍 Testing with {model_name}...")
        
        try:
            # Initialize extractor
            extractor = SignatureExtractor(model_name)
            
            # Test document processing
            result = extractor.process_document(test_path, "test_user_001")
            
            if result["success"]:
                print(f"✅ {model_name} - Processing successful!")
                print(f"   - IoU Score: {result['iou_score']:.3f}")
                print(f"   - New Signature: {result['is_new']}")
                print(f"   - Similar Signatures: {len(result['similar_signatures'])}")
                
                if result['regions']:
                    if result['regions']['stamp_region']:
                        print(f"   - Stamp detected: {result['regions']['stamp_region']}")
                    if result['regions']['signature_region']:
                        print(f"   - Signature detected: {result['regions']['signature_region']}")
                
                results[model_name] = result
            else:
                print(f"❌ {model_name} - Processing failed: {result['error']}")
                results[model_name] = result
                
        except Exception as e:
            print(f"❌ {model_name} - Error: {str(e)}")
            results[model_name] = {"success": False, "error": str(e)}
    
    return results, test_path

def test_similarity_comparison():
    """Test signature similarity comparison"""
    print("\n🔍 Testing Signature Similarity Comparison...")
    
    # Create two similar signature images
    img1 = create_test_signature_image(signature_text="John Doe")
    img2 = create_test_signature_image(signature_text="John Doe")  # Same signature
    
    # Save test images
    path1 = "test_sig1.png"
    path2 = "test_sig2.png"
    img1.save(path1)
    img2.save(path2)
    
    try:
        extractor = SignatureExtractor("Qwen/Qwen2.5-VL-7B-Instruct")
        
        # Process first signature
        result1 = extractor.process_document(path1, "user1")
        
        # Process second signature
        result2 = extractor.process_document(path2, "user2")
        
        if result1["success"] and result2["success"]:
            # Calculate similarity
            similarity = extractor.calculate_iou_similarity(
                result1["signature_data"]["features"],
                result2["signature_data"]["features"]
            )
            
            print(f"✅ Similarity test completed!")
            print(f"   - Signature 1 IoU: {result1['iou_score']:.3f}")
            print(f"   - Signature 2 IoU: {result2['iou_score']:.3f}")
            print(f"   - Cross-similarity: {similarity:.3f}")
            
            return True
        else:
            print("❌ Similarity test failed - processing errors")
            return False
            
    except Exception as e:
        print(f"❌ Similarity test error: {str(e)}")
        return False
    finally:
        # Clean up test files
        for path in [path1, path2]:
            if os.path.exists(path):
                os.remove(path)

def test_database_operations():
    """Test database operations"""
    print("\n🗄️ Testing Database Operations...")
    
    try:
        extractor = SignatureExtractor("Qwen/Qwen2.5-VL-7B-Instruct")
        
        # Create test signature data
        test_data = {
            "analysis": "Test signature analysis",
            "features": {
                "text_description": "Test signature",
                "stroke_patterns": ["curved", "flowing"],
                "letter_shapes": ["cursive"],
                "style_classification": "cursive"
            },
            "signature_region": [100, 200, 300, 250],
            "model_used": "test_model",
            "timestamp": "2024-01-15T10:00:00"
        }
        
        # Test storing signature
        extractor.store_signature("test_user", test_data, "test_image.png")
        print("✅ Signature stored successfully")
        
        # Test finding similar signatures
        similar = extractor.find_similar_signatures(test_data["features"])
        print(f"✅ Found {len(similar)} similar signatures")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test error: {str(e)}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 Starting Comprehensive Test Suite")
    print("=" * 50)
    
    test_results = {
        "extraction": False,
        "similarity": False,
        "database": False
    }
    
    # Test 1: Signature Extraction
    try:
        extraction_results, test_path = test_signature_extraction()
        test_results["extraction"] = any(r.get("success", False) for r in extraction_results.values())
    except Exception as e:
        print(f"❌ Extraction test failed: {str(e)}")
    
    # Test 2: Similarity Comparison
    try:
        test_results["similarity"] = test_similarity_comparison()
    except Exception as e:
        print(f"❌ Similarity test failed: {str(e)}")
    
    # Test 3: Database Operations
    try:
        test_results["database"] = test_database_operations()
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   - Signature Extraction: {'✅ PASS' if test_results['extraction'] else '❌ FAIL'}")
    print(f"   - Similarity Comparison: {'✅ PASS' if test_results['similarity'] else '❌ FAIL'}")
    print(f"   - Database Operations: {'✅ PASS' if test_results['database'] else '❌ FAIL'}")
    
    overall_success = all(test_results.values())
    print(f"\n🎯 Overall Test Status: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    # Clean up
    if 'test_path' in locals() and os.path.exists(test_path):
        os.remove(test_path)
    
    return overall_success

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)