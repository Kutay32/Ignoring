# VLM Signature Detector

A comprehensive signature detection and comparison module that combines **YOLOv8s.pt** for object detection with **Vision-Language Model (VLM)** capabilities for VLM-style signature analysis and comparison.

## Features

### 🔍 **YOLOv8s.pt Integration**
- Accurate signature region detection using YOLOv8s.pt model
- Configurable confidence thresholds
- Multiple detection candidates with metadata
- Optimized for signature-like regions (aspect ratio, area filtering)

### 🧠 **VLM Analysis**
- Deep signature structure analysis using Qwen2-VL models
- VLM-style feature extraction for signature characteristics
- Hierarchical, modular, and sequential signature architecture detection
- Token-level and sequence pattern analysis

### 🔄 **Advanced Comparison**
- Multi-dimensional similarity scoring
- Text similarity using TF-IDF
- Structural component matching
- VLM token similarity
- Pattern and architecture comparison

### 💾 **Database Management**
- SQLite database for signature storage
- Comprehensive metadata tracking
- Comparison history storage
- Statistics and analytics

## Installation

### Prerequisites
```bash
pip install torch torchvision
pip install ultralytics
pip install transformers accelerate
pip install opencv-python
pip install scikit-learn
pip install Pillow
pip install matplotlib seaborn
```

### Model Weights
Ensure you have `yolov8s.pt` in the `weights/` directory:
```
workspace/
├── weights/
│   └── yolov8s.pt
├── vlm_signature_detector.py
├── test_vlm_signature_detector.py
└── example_vlm_usage.py
```

## Quick Start

### Basic Usage

```python
from vlm_signature_detector import VLMSignatureDetector

# Initialize detector
detector = VLMSignatureDetector(
    yolo_model_path="weights/yolov8s.pt",
    vlm_model_name="Qwen/Qwen2-VL-2B-Instruct",
    confidence_threshold=0.5
)

# Process a document
result = detector.process_document_vlm("document.jpg", user_id="user_001")

if result["success"]:
    print(f"Detected {result['total_detections']} signature regions")
    print(f"Performed {result['total_vlm_analyses']} VLM analyses")
    print(f"Stored {len(result['stored_signatures'])} new signatures")
```

### Advanced Usage

```python
# Detect signature regions with YOLO
regions = detector.detect_signature_regions_yolo("document.jpg")

# Analyze signature structure with VLM
for region in regions:
    vlm_analysis = detector.analyze_signature_structure_vlm(
        "document.jpg", region['bbox']
    )
    print(f"VLM Confidence: {vlm_analysis['vlm_confidence']:.3f}")

# Compare signatures
comparison = detector.compare_signatures_vlm(signature_id1, signature_id2)
print(f"Verdict: {comparison['verdict']}")
print(f"Confidence: {comparison['confidence']:.3f}")

# Find similar signatures
similar = detector.find_similar_signatures_vlm(query_features, threshold=0.7)
```

## API Reference

### VLMSignatureDetector Class

#### Constructor
```python
VLMSignatureDetector(
    yolo_model_path: str = "weights/yolov8s.pt",
    vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    confidence_threshold: float = 0.5
)
```

#### Key Methods

##### `detect_signature_regions_yolo(image_path: str) -> List[Dict]`
Detects signature regions using YOLOv8s.pt
- **Returns**: List of detected regions with bounding boxes, confidence scores, and metadata

##### `analyze_signature_structure_vlm(image_path: str, signature_region: List[int] = None) -> Dict`
Analyzes signature structure using VLM
- **Returns**: VLM analysis with structure features and confidence scores

##### `process_document_vlm(image_path: str, user_id: str = None) -> Dict`
Complete processing pipeline
- **Returns**: Comprehensive results including detections, analyses, and storage

##### `compare_signatures_vlm(signature1_id: int, signature2_id: int) -> Dict`
Compares two signatures using VLM analysis
- **Returns**: Detailed comparison results with similarity scores

##### `find_similar_signatures_vlm(query_features: Dict, threshold: float = 0.7) -> List[Dict]`
Finds similar signatures in database
- **Returns**: List of similar signatures with similarity scores

## VLM Analysis Features

### Structural Analysis
- **Hierarchical**: Signature with clear hierarchical structure
- **Modular**: Signature with distinct modular components
- **Sequential**: Signature with linear sequential flow
- **Mixed**: Combination of structural patterns

### VLM-Specific Features
- **Token-level characteristics**: Individual letter structures
- **Sequence patterns**: Letter combinations and transitions
- **Contextual relationships**: Component interactions
- **Semantic structure**: Meaning representation

### Comparative Features
- **Structural components**: Hierarchical, layered, modular, etc.
- **Spatial relationships**: Above, below, adjacent, overlapping
- **Pattern features**: Repetitive, unique, consistent, flowing
- **Complexity metrics**: High, medium, low complexity levels

## Database Schema

### vlm_signatures Table
- `id`: Primary key
- `user_id`: User identifier
- `signature_hash`: Unique signature hash
- `signature_data`: Complete signature data (JSON)
- `vlm_analysis`: VLM analysis text
- `yolo_detection`: YOLO detection metadata (JSON)
- `signature_features`: Extracted features (JSON)
- `embedding_vector`: Feature embedding (JSON)
- `image_path`: Source image path
- `timestamp`: Creation timestamp
- `yolo_confidence`: YOLO detection confidence
- `vlm_confidence`: VLM analysis confidence
- `signature_type`: Signature architecture type
- `structure_type`: Structural classification

### signature_comparisons Table
- `id`: Primary key
- `signature1_id`: First signature ID
- `signature2_id`: Second signature ID
- `yolo_similarity`: YOLO-based similarity
- `vlm_similarity`: VLM-based similarity
- `overall_similarity`: Combined similarity score
- `comparison_method`: Method used for comparison
- `timestamp`: Comparison timestamp

## Testing

### Run Test Suite
```bash
python test_vlm_signature_detector.py
```

The test suite includes:
- YOLO detection testing
- VLM analysis testing
- Signature comparison testing
- Database operations testing
- Full pipeline testing

### Run Example
```bash
python example_vlm_usage.py
```

## Configuration

### YOLO Configuration
- **Model**: YOLOv8s.pt (small, fast, accurate)
- **Confidence threshold**: 0.5 (adjustable)
- **Detection filtering**: Area > 1000, aspect ratio 0.2-5.0

### VLM Configuration
- **Model**: Qwen2-VL-2B-Instruct (memory efficient)
- **Fallback**: Automatic fallback to smaller models if memory insufficient
- **Analysis depth**: Comprehensive structural analysis
- **Token limit**: 2048 tokens for detailed analysis

### Similarity Thresholds
- **Default threshold**: 0.7 for similarity matching
- **Match verdict**: > 0.8 for signature match
- **Weighted scoring**: Text (30%), Structural (25%), VLM tokens (20%), Pattern (15%), Architecture (10%)

## Performance Considerations

### Memory Usage
- **YOLOv8s.pt**: ~22MB model size
- **Qwen2-VL-2B**: ~4GB VRAM (with optimizations)
- **CPU fallback**: Available for systems without GPU

### Processing Speed
- **YOLO detection**: ~100-500ms per image
- **VLM analysis**: ~2-10 seconds per signature (depending on hardware)
- **Database operations**: <100ms for most operations

### Optimization Tips
1. Use GPU acceleration when available
2. Adjust confidence thresholds based on use case
3. Batch process multiple images when possible
4. Use appropriate VLM model size for your hardware

## Troubleshooting

### Common Issues

#### YOLO Model Not Found
```
Error: YOLOv8s.pt not found at weights/yolov8s.pt
```
**Solution**: Ensure the model file is in the `weights/` directory

#### VLM Model Loading Failed
```
Error loading VLM model: CUDA out of memory
```
**Solution**: The system will automatically fallback to a smaller model

#### Low Detection Accuracy
**Solutions**:
- Adjust confidence threshold
- Check image quality and resolution
- Ensure signatures are clearly visible
- Consider image preprocessing

### Debug Mode
Enable detailed logging by setting environment variable:
```bash
export VLM_DEBUG=1
python your_script.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Ultralytics** for YOLOv8 implementation
- **Qwen Team** for Vision-Language Model capabilities
- **OpenCV** for computer vision utilities
- **Transformers** for VLM model integration