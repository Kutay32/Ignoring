#!/usr/bin/env python3
"""
Example usage of VLM Signature Detector
Demonstrates how to use the YOLOv8s.pt + VLM signature detection module
"""

from vlm_signature_detector import VLMSignatureDetector
import os

def main():
    """Example usage of VLM Signature Detector"""
    
    print("VLM Signature Detector - Usage Example")
    print("="*50)
    
    # Initialize the detector
    print("Initializing VLM Signature Detector...")
    detector = VLMSignatureDetector(
        yolo_model_path="weights/yolov8s.pt",
        vlm_model_name="Qwen/Qwen2-VL-2B-Instruct",
        confidence_threshold=0.5
    )
    print("✓ Detector initialized successfully!")
    
    # Example 1: Process a document image
    print("\n" + "="*30)
    print("Example 1: Process Document Image")
    print("="*30)
    
    # Replace with your actual image path
    image_path = "path/to/your/document.jpg"
    
    if os.path.exists(image_path):
        print(f"Processing document: {image_path}")
        
        # Process the document
        result = detector.process_document_vlm(image_path, user_id="user_001")
        
        if result["success"]:
            print(f"✓ Successfully processed document")
            print(f"  - YOLO detections: {result['total_detections']}")
            print(f"  - VLM analyses: {result['total_vlm_analyses']}")
            print(f"  - Stored signatures: {len(result['stored_signatures'])}")
            print(f"  - Similar signatures found: {len(result['similar_signatures'])}")
            
            # Show details of detected signatures
            for i, analysis in enumerate(result["vlm_analyses"]):
                print(f"\nSignature {i+1}:")
                print(f"  - YOLO confidence: {analysis.get('confidence', 0):.3f}")
                print(f"  - VLM confidence: {analysis.get('vlm_confidence', 0):.3f}")
                print(f"  - Combined confidence: {analysis.get('combined_confidence', 0):.3f}")
                
                features = analysis.get('structure_features', {})
                print(f"  - Architecture: {features.get('signature_architecture', 'unknown')}")
                print(f"  - Structural components: {features.get('structural_components', [])}")
        else:
            print(f"✗ Processing failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path to test the detector")
    
    # Example 2: Compare two signatures
    print("\n" + "="*30)
    print("Example 2: Compare Signatures")
    print("="*30)
    
    # This would work if you have signatures stored in the database
    try:
        # Get some statistics first
        stats = detector.get_signature_statistics()
        print(f"Database contains {stats.get('total_signatures', 0)} signatures")
        
        if stats.get('total_signatures', 0) >= 2:
            # Compare first two signatures (this is just an example)
            comparison = detector.compare_signatures_vlm(1, 2)
            
            if "error" not in comparison:
                print(f"✓ Signature comparison completed")
                print(f"  - Verdict: {comparison['verdict']}")
                print(f"  - Confidence: {comparison['confidence']:.3f}")
                print(f"  - Text similarity: {comparison['similarities'].get('text_similarity', 0):.3f}")
                print(f"  - Structural similarity: {comparison['similarities'].get('structural_similarity', 0):.3f}")
            else:
                print(f"✗ Comparison failed: {comparison['error']}")
        else:
            print("Need at least 2 signatures in database for comparison")
    except Exception as e:
        print(f"Comparison example failed: {e}")
    
    # Example 3: Find similar signatures
    print("\n" + "="*30)
    print("Example 3: Find Similar Signatures")
    print("="*30)
    
    try:
        # This would work if you have signatures in the database
        stats = detector.get_signature_statistics()
        
        if stats.get('total_signatures', 0) > 0:
            # Create a sample query feature set
            sample_features = {
                "text_analysis": "Sample signature analysis",
                "structural_components": ["hierarchical", "modular"],
                "vlm_tokens": ["token", "sequence", "pattern"],
                "signature_architecture": "hierarchical",
                "pattern_features": ["consistent", "flowing"],
                "semantic_structure": ["meaningful", "expressive"]
            }
            
            similar_signatures = detector.find_similar_signatures_vlm(sample_features, threshold=0.5)
            
            print(f"✓ Found {len(similar_signatures)} similar signatures")
            for i, sig in enumerate(similar_signatures[:3]):  # Show top 3
                print(f"  {i+1}. ID: {sig['id']}, Score: {sig['overall_score']:.3f}")
        else:
            print("No signatures in database to search")
    except Exception as e:
        print(f"Similarity search failed: {e}")
    
    print("\n" + "="*50)
    print("Example completed!")
    print("="*50)
    
    print("\nKey Features of VLM Signature Detector:")
    print("• YOLOv8s.pt for accurate signature region detection")
    print("• VLM analysis for signature structure understanding")
    print("• VLM-style similarity comparison")
    print("• Database storage for signature management")
    print("• Comprehensive confidence scoring")
    print("• Support for multiple signature types and styles")

if __name__ == "__main__":
    main()