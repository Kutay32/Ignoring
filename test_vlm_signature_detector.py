#!/usr/bin/env python3
"""
Test script for VLM Signature Detector module
Tests YOLOv8s.pt integration with VLM analysis for signature detection and comparison
"""

import os
import sys
import time
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from vlm_signature_detector import VLMSignatureDetector

def create_test_signature_image(filename: str, signature_text: str = "John Doe", 
                               signature_style: str = "cursive") -> str:
    """
    Create a test signature image for testing purposes
    
    Args:
        filename: Output filename for the test image
        signature_text: Text to use as signature
        signature_style: Style of signature (cursive, print, artistic)
    
    Returns:
        Path to created image
    """
    # Create a white background
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Add some document-like content
    try:
        # Try to use a default font
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    except:
        font_large = None
        font_small = None
    
    # Add document header
    draw.text((50, 50), "OFFICIAL DOCUMENT", fill='black', font=font_large)
    draw.text((50, 100), "Date: 2024-01-15", fill='black', font=font_small)
    draw.text((50, 130), "Document ID: DOC-2024-001", fill='black', font=font_small)
    
    # Add some content
    content_lines = [
        "This is a test document for signature verification.",
        "The signature below should be detected and analyzed.",
        "Please sign in the designated area.",
        "",
        "Terms and conditions apply.",
        "This document is legally binding."
    ]
    
    y_offset = 200
    for line in content_lines:
        draw.text((50, y_offset), line, fill='black', font=font_small)
        y_offset += 30
    
    # Add signature area
    signature_x, signature_y = 500, 450
    signature_width, signature_height = 200, 80
    
    # Draw signature box
    draw.rectangle([signature_x, signature_y, signature_x + signature_width, signature_y + signature_height], 
                   outline='black', width=2)
    draw.text((signature_x + 10, signature_y + 10), "Signature:", fill='black', font=font_small)
    
    # Add signature based on style
    if signature_style == "cursive":
        # Create a more cursive-like signature
        points = []
        start_x, start_y = signature_x + 20, signature_y + 40
        for i, char in enumerate(signature_text):
            x = start_x + i * 15 + np.random.randint(-5, 5)
            y = start_y + np.random.randint(-10, 10)
            points.append((x, y))
        
        # Draw connected signature
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill='black', width=3)
    elif signature_style == "artistic":
        # Create an artistic signature with flourishes
        start_x, start_y = signature_x + 20, signature_y + 40
        for i, char in enumerate(signature_text):
            x = start_x + i * 20
            y = start_y + np.sin(i * 0.5) * 10
            draw.text((x, y), char, fill='black', font=font_large)
            # Add flourishes
            if i % 2 == 0:
                draw.ellipse([x-5, y-5, x+15, y+15], outline='black', width=1)
    else:  # print style
        start_x, start_y = signature_x + 20, signature_y + 40
        draw.text((start_x, start_y), signature_text, fill='black', font=font_large)
    
    # Save the image
    image.save(filename)
    print(f"Created test signature image: {filename}")
    return filename

def test_yolo_detection(detector: VLMSignatureDetector, image_path: str):
    """Test YOLO detection functionality"""
    print("\n" + "="*50)
    print("Testing YOLO Detection")
    print("="*50)
    
    try:
        regions = detector.detect_signature_regions_yolo(image_path)
        print(f"YOLO detected {len(regions)} potential signature regions")
        
        for i, region in enumerate(regions):
            print(f"\nRegion {i+1}:")
            print(f"  Bounding box: {region['bbox']}")
            print(f"  Confidence: {region['confidence']:.3f}")
            print(f"  Area: {region['area']}")
            print(f"  Aspect ratio: {region['aspect_ratio']:.3f}")
            print(f"  Method: {region['method']}")
        
        return regions
    except Exception as e:
        print(f"YOLO detection failed: {e}")
        return []

def test_vlm_analysis(detector: VLMSignatureDetector, image_path: str, regions: list):
    """Test VLM analysis functionality"""
    print("\n" + "="*50)
    print("Testing VLM Analysis")
    print("="*50)
    
    try:
        vlm_results = []
        
        if regions:
            # Test with detected regions
            for i, region in enumerate(regions[:2]):  # Test first 2 regions
                print(f"\nAnalyzing region {i+1} with VLM...")
                vlm_analysis = detector.analyze_signature_structure_vlm(
                    image_path, region['bbox']
                )
                vlm_results.append(vlm_analysis)
                
                print(f"VLM Confidence: {vlm_analysis.get('vlm_confidence', 0):.3f}")
                print(f"Structure Features: {len(vlm_analysis.get('structure_features', {}))}")
                
                # Print some key features
                features = vlm_analysis.get('structure_features', {})
                print(f"Signature Architecture: {features.get('signature_architecture', 'unknown')}")
                print(f"Structural Components: {features.get('structural_components', [])}")
                print(f"VLM Tokens: {features.get('vlm_tokens', [])}")
        else:
            # Test without specific region (full image)
            print("Analyzing full image with VLM...")
            vlm_analysis = detector.analyze_signature_structure_vlm(image_path)
            vlm_results.append(vlm_analysis)
            
            print(f"VLM Confidence: {vlm_analysis.get('vlm_confidence', 0):.3f}")
            features = vlm_analysis.get('structure_features', {})
            print(f"Signature Architecture: {features.get('signature_architecture', 'unknown')}")
        
        return vlm_results
    except Exception as e:
        print(f"VLM analysis failed: {e}")
        return []

def test_signature_comparison(detector: VLMSignatureDetector, vlm_results: list):
    """Test signature comparison functionality"""
    print("\n" + "="*50)
    print("Testing Signature Comparison")
    print("="*50)
    
    try:
        if len(vlm_results) >= 2:
            features1 = vlm_results[0].get('structure_features', {})
            features2 = vlm_results[1].get('structure_features', {})
            
            similarities = detector.calculate_vlm_similarity(features1, features2)
            
            print("VLM Similarity Results:")
            for key, value in similarities.items():
                print(f"  {key}: {value:.3f}")
            
            return similarities
        else:
            print("Need at least 2 VLM results for comparison")
            return {}
    except Exception as e:
        print(f"Signature comparison failed: {e}")
        return {}

def test_database_operations(detector: VLMSignatureDetector, vlm_results: list, image_path: str):
    """Test database storage and retrieval"""
    print("\n" + "="*50)
    print("Testing Database Operations")
    print("="*50)
    
    try:
        # Store signatures
        stored_ids = []
        for i, result in enumerate(vlm_results):
            signature_id = detector.store_signature_vlm(
                f"test_user_{i+1}", result, image_path
            )
            if signature_id:
                stored_ids.append(signature_id)
                print(f"Stored signature {i+1} with ID: {signature_id}")
        
        # Test finding similar signatures
        if vlm_results and stored_ids:
            query_features = vlm_results[0].get('structure_features', {})
            similar_signatures = detector.find_similar_signatures_vlm(query_features)
            
            print(f"Found {len(similar_signatures)} similar signatures")
            for sig in similar_signatures[:3]:  # Show top 3
                print(f"  ID: {sig['id']}, Score: {sig['overall_score']:.3f}")
        
        # Test signature comparison
        if len(stored_ids) >= 2:
            comparison = detector.compare_signatures_vlm(stored_ids[0], stored_ids[1])
            if "error" not in comparison:
                print(f"Signature comparison result: {comparison['verdict']}")
                print(f"Confidence: {comparison['confidence']:.3f}")
        
        # Get statistics
        stats = detector.get_signature_statistics()
        print(f"Database statistics: {stats}")
        
        return stored_ids
    except Exception as e:
        print(f"Database operations failed: {e}")
        return []

def test_full_pipeline(detector: VLMSignatureDetector, image_path: str):
    """Test the complete VLM processing pipeline"""
    print("\n" + "="*50)
    print("Testing Full VLM Pipeline")
    print("="*50)
    
    try:
        result = detector.process_document_vlm(image_path, "test_user_pipeline")
        
        print(f"Pipeline success: {result['success']}")
        print(f"YOLO detections: {result['total_detections']}")
        print(f"VLM analyses: {result['total_vlm_analyses']}")
        print(f"Stored signatures: {len(result['stored_signatures'])}")
        print(f"Similar signatures found: {len(result['similar_signatures'])}")
        
        if result['success']:
            print("\nPipeline completed successfully!")
        else:
            print(f"Pipeline failed: {result.get('error', 'Unknown error')}")
        
        return result
    except Exception as e:
        print(f"Full pipeline test failed: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Main test function"""
    print("VLM Signature Detector Test Suite")
    print("="*50)
    
    # Check if YOLOv8s.pt exists
    yolo_path = "weights/yolov8s.pt"
    if not os.path.exists(yolo_path):
        print(f"Error: YOLOv8s.pt not found at {yolo_path}")
        print("Please ensure the weights file is in the weights/ directory")
        return
    
    # Create test images
    test_images = []
    
    # Create different signature styles for testing
    styles = ["cursive", "print", "artistic"]
    for i, style in enumerate(styles):
        filename = f"test_signature_{style}_{i+1}.png"
        create_test_signature_image(filename, f"Test User {i+1}", style)
        test_images.append(filename)
    
    try:
        # Initialize detector
        print("Initializing VLM Signature Detector...")
        detector = VLMSignatureDetector(
            yolo_model_path=yolo_path,
            vlm_model_name="Qwen/Qwen2-VL-2B-Instruct",
            confidence_threshold=0.3
        )
        print("Detector initialized successfully!")
        
        # Test with first image
        test_image = test_images[0]
        print(f"\nTesting with image: {test_image}")
        
        # Test individual components
        regions = test_yolo_detection(detector, test_image)
        vlm_results = test_vlm_analysis(detector, test_image, regions)
        similarities = test_signature_comparison(detector, vlm_results)
        stored_ids = test_database_operations(detector, vlm_results, test_image)
        
        # Test full pipeline
        pipeline_result = test_full_pipeline(detector, test_image)
        
        # Test with additional images for comparison
        if len(test_images) > 1:
            print(f"\nTesting with additional image: {test_images[1]}")
            additional_result = test_full_pipeline(detector, test_images[1])
        
        print("\n" + "="*50)
        print("Test Suite Completed")
        print("="*50)
        
        # Cleanup test images
        print("\nCleaning up test images...")
        for image in test_images:
            try:
                os.remove(image)
                print(f"Removed: {image}")
            except:
                pass
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup test images
        for image in test_images:
            try:
                if os.path.exists(image):
                    os.remove(image)
            except:
                pass

if __name__ == "__main__":
    main()