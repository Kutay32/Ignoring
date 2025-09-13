# Advanced Signature Detection UI - Usage Guide

## 🚀 Quick Start

Your signature detection system now uses **YOLOv8 + Qwen** integration! Here's how to use it:

### 1. Launch the UI

### Option 1: Fixed Launcher (Recommended for localhost issues)
```bash
# From the Ignoring folder
python fix_ui_launcher.py
```
This launcher includes server monitoring, automatic browser opening, and better error handling.

### Option 2: Original Launcher
```bash
python run_advanced_ui.py
```

### Option 3: CPU-Only Version (if GPU memory is insufficient)
```bash
python cpu_ui_launcher.py
```

### Option 4: Direct Launch
```bash
python advanced_gradio_ui.py
```

### Option 5: CUDA Test (Check if GPU works)
```bash
python cuda_test.py
```

### Option 6: Direct Qwen GPU Test (Verify Qwen works on GPU)
```bash
python run_qwen_gpu.py
```

### Option 7: Simple UI Test (Check if UI loads)
```bash
python simple_ui_test.py
```

This launches a basic test interface on port 7862 to verify Gradio works.

### 2. Load Your Models

1. The Qwen model is pre-selected:
   - `Qwen/Qwen2.5-VL-7B-Instruct` (only option available)

2. Click **"Load Model"**

3. You'll see confirmation that both models loaded:
   ```
   ✅ Models loaded successfully!
     • Qwen: Qwen/Qwen2.5-VL-7B-Instruct
     • YOLOv8: yolov8s.pt
   ```

## 🔍 How It Works

### Architecture Overview

```
Document Image → YOLOv8 Detection → Qwen Analysis → Database Storage
     ↓               ↓              ↓              ↓
  Input          Find Regions   Extract Features  Store Results
```

- **YOLOv8 (yolov8s.pt)**: Detects potential signature regions using object detection
- **Qwen VLM**: Analyzes detected regions for detailed signature features
- **OpenCV Fallback**: Automatically falls back to traditional detection if YOLO fails
- **Database**: Stores all signatures with advanced similarity metrics

### Detection Methods

The system tries YOLOv8 first, then falls back to OpenCV if needed:

1. **YOLOv8 Detection**: Uses object detection to find signature-like regions
2. **OpenCV Fallback**: Uses edge detection, color analysis, and template matching

## 📱 Using the Interface

### Signature Detection Tab

1. **Upload Image**: Drag & drop or browse for document images
2. **User ID** (optional): Enter user ID for signature storage
3. **Similarity Threshold**: Adjust sensitivity (0.1-1.0, default 0.7)
4. **Process Image**: Click to analyze

#### Results Display

- **Processing Summary**: Overview of detected regions and methods used
- **Detailed Analysis**: Qwen's comprehensive signature analysis
- **Similarity Report**: Matching signatures from database

### Signature Comparison Tab

Compare any two signatures by their database IDs:

1. Enter Signature 1 ID and Signature 2 ID
2. Click "Compare Signatures"
3. View detailed similarity analysis

### Database Management Tab

View database statistics including:
- Total signatures stored
- Signature types distribution
- Recent signatures

## 🎯 Key Features

### Hardware Acceleration
- **CUDA GPU Support**: Automatic detection and utilization of NVIDIA GPUs
- **8-bit Quantization**: Reduces memory usage by ~50% for large models
- **CPU Fallback**: Automatically switches to CPU if GPU memory is insufficient
- **Memory Optimization**: Smart memory management for stable operation

### Advanced Detection
- **YOLOv8 Integration**: State-of-the-art object detection for signatures
- **Multi-Method Approach**: Combines AI detection with traditional computer vision
- **Smart Fallback**: Never fails - automatically switches detection methods

### Comprehensive Analysis
- **Qwen VLM Analysis**: Detailed signature feature extraction
- **Multiple Similarity Metrics**: Text, feature, and overall similarity scores
- **Signature Classification**: Type, complexity, legibility, and style analysis

### Database Features
- **Advanced Storage**: SQLite database with comprehensive metadata
- **Similarity Search**: Find matching signatures across the database
- **Comparison Reports**: Detailed side-by-side signature analysis

## 🔧 Configuration

### Model Selection

Your system is configured to use:

- **Qwen2.5-VL-7B**: Advanced analysis quality with 8-bit quantization for memory efficiency

### YOLOv8 Configuration

The system automatically uses `weights/yolov8s.pt`. The model is configured to:
- Detect objects with confidence > 0.3
- Filter for signature-like aspect ratios (0.5-5.0)
- Focus on reasonable image regions (0.1%-30% of total area)

## 📊 Understanding Results

### Detection Results
- **Method**: Shows whether YOLOv8 or OpenCV was used
- **Confidence**: Detection confidence score
- **Area**: Size of detected region in pixels
- **Aspect Ratio**: Width-to-height ratio

### Signature Analysis
- **Type**: cursive, print, mixed, artistic
- **Complexity**: simple, moderate, complex
- **Size**: small, medium, large
- **Legibility**: high, medium, low

### Similarity Scores
- **Overall Similarity**: Combined text and feature similarity (0-1)
- **Text Similarity**: VLM analysis text comparison
- **Feature Similarity**: Structured feature matching

## 🔍 System Diagnostics

### CUDA Status Check
Run the CUDA test to verify your GPU setup:
```bash
python cuda_test.py
```

**Expected Results for Your System:**
- ✅ CUDA Available: True
- ✅ GPU: NVIDIA GeForce RTX 3080 Ti (12.0GB)
- ✅ PyTorch CUDA: 12.9
- ✅ GPU tensor operations: Working
- ✅ Transformers CUDA: Working
- ✅ YOLO CUDA: Working
- ✅ Qwen2.5-VL-7B: Working with 8-bit quantization

**If CUDA test fails:**
- Try: `python cpu_ui_launcher.py` (CPU-only mode)
- Check GPU drivers: NVIDIA Control Panel → System Information
- Update PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129`

## 🚨 Troubleshooting

### UI Appears Blank or Won't Load
If you see a blank screen or the UI doesn't load properly:

1. **Try the Simple UI Test**:
   ```bash
   python simple_ui_test.py
   ```
   This loads a basic interface on port 7862 to test if Gradio works.

2. **Check Browser and Cache**:
   - Try a different browser (Chrome, Firefox, Edge)
   - Clear browser cache and cookies
   - Disable browser extensions temporarily

3. **Check Firewall/Antivirus**:
   - Temporarily disable firewall/antivirus
   - Some security software blocks local servers

4. **Manual Browser Access**:
   - Open browser manually and go to: `http://localhost:7860`
   - Or try: `http://127.0.0.1:7860`

5. **Debug Mode**:
   ```bash
   python debug_ui_test.py
   ```
   This runs comprehensive tests to identify the issue.

### UI Crashes During Startup
If the UI closes suddenly during model loading:

1. **Try the Stable Launcher** (recommended):
   ```bash
   python stable_ui_launcher.py
   ```
   This launcher has better error handling and system checks.

2. **Check Memory Usage**:
   - Close other applications using GPU memory
   - Restart your computer
   - Try the CPU version: `python cpu_ui_launcher.py`

3. **Monitor Loading Progress**:
   - The model loading shows progress for each of the 5 shards
   - Loading takes 1-3 minutes depending on your internet speed
   - Don't close the terminal while "Loading checkpoint shards" is shown

### Model Loading Issues
- Ensure `weights/yolov8s.pt` exists
- Check internet connection for Qwen model download
- Verify CUDA installation for GPU acceleration
- If CUDA runs out of memory, the system automatically falls back to CPU

### Detection Problems
- If YOLO fails, the system automatically uses OpenCV
- Try adjusting similarity threshold
- Ensure images are clear and well-lit

### Performance Issues
- The system automatically uses 8-bit quantization to reduce memory usage
- If memory issues persist, consider upgrading hardware or using CPU mode
- Reduce image resolution for faster processing
- GPU acceleration provides best performance when available

### Common Error Messages

**"CUDA out of memory"**
- Close other GPU-intensive applications
- Try CPU version: `python cpu_ui_launcher.py`
- Restart your computer

**"Loading checkpoint shards" stuck**
- Check your internet connection
- Wait patiently (can take several minutes)
- Don't close the terminal

**"Model loaded with 8-bit quantization!"**
- This is normal! It means memory optimization is working

## 📁 File Structure

```
Signature Detection/
├── Ignoring/
│   ├── advanced_gradio_ui.py      # Updated UI with YOLO integration
│   ├── advanced_signature_detector.py  # Core detector with YOLO + Qwen
│   ├── run_advanced_ui.py         # Launch script
│   ├── weights/
│   │   └── yolov8s.pt            # YOLOv8 weights
│   └── advanced_signatures.db     # Signature database
```

## 🎉 What's New

Compared to the previous version:

- ✅ **YOLOv8 Integration**: Advanced object detection for signatures
- ✅ **Improved Accuracy**: Better signature region detection
- ✅ **Smart Fallback**: Never fails detection
- ✅ **Enhanced UI**: Clear indication of detection methods used
- ✅ **Robust Architecture**: Automatic method switching

Your signature detection system is now more powerful and reliable than ever!
