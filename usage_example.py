#!/usr/bin/env python3
"""
Usage example for the modified Advanced Signature Detector
Now using YOLOv8 for signature detection and Qwen for analysis
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from advanced_signature_detector import AdvancedSignatureDetector

def main():
    """Example usage of the signature detector with YOLO + Qwen"""

    # Initialize the detector with YOLO weights
    print("Initializing Advanced Signature Detector...")
    print("Using YOLOv8 (yolov8s.pt) for signature detection")
    print("Using Qwen (Qwen2-VL-2B-Instruct) for signature analysis")

    detector = AdvancedSignatureDetector(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        yolo_weights_path="weights/yolov8s.pt"
    )

    # Example image path (replace with your actual image)
    # For testing, you can use any document image containing signatures
    image_path = "path/to/your/document.jpg"  # Replace with actual path

    if not os.path.exists(image_path):
        print(f"Example image not found: {image_path}")
        print("To use this example:")
        print("1. Place a document image with signatures in the Ignoring folder")
        print("2. Update the image_path variable above")
        print("3. Run this script again")
        return

    print(f"\nProcessing document: {image_path}")

    # Process the document
    results = detector.process_document_advanced(image_path, user_id="example_user")

    if results["success"]:
        print("✓ Processing completed successfully!")
        print(f"  - Detected regions: {results['total_signatures_detected']}")
        print(f"  - New signatures: {results['new_signatures_count']}")
        print(f"  - Matched signatures: {results['matched_signatures_count']}")

        # Show details of detected signatures
        for i, sig_data in enumerate(results['signature_results'][:3]):  # Show first 3
            print(f"\n  Signature {i+1}:")
            print(f"    Type: {sig_data['features']['signature_type']}")
            print(f"    Confidence: {sig_data['confidence_score']:.3f}")
            print(f"    Detection method: {sig_data['detection_method']}")

        # Show similar signatures found
        if results['similar_signatures']:
            print(f"\n  Found {len(results['similar_signatures'])} similar signatures in database")
            for sim in results['similar_signatures'][:2]:  # Show top 2
                print(f"    User: {sim['user_id']}, Similarity: {sim['overall_score']:.3f}")

    else:
        print(f"✗ Processing failed: {results['error']}")

    print("\n" + "="*50)
    print("System Configuration:")
    print("- Signature Detection: YOLOv8 (yolov8s.pt)")
    print("- Feature Analysis: Qwen2-VL-2B-Instruct")
    print("- Fallback: OpenCV detection if YOLO fails")
    print("="*50)

if __name__ == "__main__":
    main()
