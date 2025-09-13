#!/usr/bin/env python3
"""
Super Fast YOLO-Only Signature Detection UI
No LLM, just YOLO detection - processes in 2-3 seconds!
"""

import sys
import os
import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn
from PIL import Image
from advanced_signature_detector import AdvancedSignatureDetector

class YOLOUIServer:
    def __init__(self):
        self.app = FastAPI(
            title="YOLO Signature Detection",
            description="Super fast signature detection using only YOLO"
        )
        self.detector = None
        self.setup_routes()
        self.initialize_detector()

    def initialize_detector(self):
        """Initialize YOLO detector"""
        try:
            self.detector = AdvancedSignatureDetector()
            self.detector._load_yolo_model()
            if self.detector.yolo_model:
                print("✅ YOLO model loaded successfully")
            else:
                print("❌ YOLO model failed to load")
        except Exception as e:
            print(f"❌ Error initializing detector: {e}")

    def setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def serve_ui():
            """Serve the main UI"""
            return HTMLResponse(content=self.get_html())

        @self.app.post("/api/detect")
        async def detect_signatures(file: UploadFile = File(...)):
            """Detect signatures in uploaded image"""
            if not self.detector or not self.detector.yolo_model:
                return {
                    "success": False,
                    "error": "YOLO model not loaded"
                }

            try:
                # Read and save image temporarily
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))

                temp_path = f"/tmp/yolo_detect_{os.getpid()}_{hash(file.filename)}.png"
                image.save(temp_path)

                # Run YOLO detection
                import time
                start_time = time.time()
                regions = self.detector.detect_signature_regions_yolo(temp_path)
                processing_time = time.time() - start_time

                # Convert image to base64 for display
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                return {
                    "success": True,
                    "image": f"data:image/png;base64,{img_str}",
                    "signatures_found": len(regions),
                    "processing_time": ".2f",
                    "regions": [
                        {
                            "id": i + 1,
                            "bbox": region["bbox"],
                            "confidence": float(region["confidence"]),
                            "area": region.get("area", 0),
                            "method": region.get("method", "yolo")
                        }
                        for i, region in enumerate(regions)
                    ]
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Processing failed: {str(e)}"
                }

    def get_html(self):
        """Generate the HTML UI"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ Fast YOLO Signature Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }

        .header h1 {
            font-size: 2.2rem;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .main-content {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }

        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 50px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 30px;
        }

        .upload-area:hover {
            background: #f0f2ff;
            border-color: #5a6fd8;
        }

        .upload-area.dragover {
            background: #e8f0ff;
            border-color: #4c63d2;
        }

        .upload-area.uploading {
            background: #fff3cd;
            border-color: #ffc107;
        }

        .upload-area.success {
            background: #d4edda;
            border-color: #28a745;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .results {
            display: none;
            margin-top: 30px;
        }

        .results.show {
            display: block;
        }

        .image-container {
            text-align: center;
            margin-bottom: 30px;
        }

        .processed-image {
            max-width: 100%;
            max-height: 500px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #667eea;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }

        .stat-label {
            color: #666;
            font-size: 0.9rem;
        }

        .regions-list {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
        }

        .region-item {
            background: white;
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 10px;
            align-items: center;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .file-input {
            display: none;
        }

        .performance-badge {
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
            margin-left: 10px;
        }

        .info-box {
            background: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
        }

        .info-box h4 {
            color: #1976d2;
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Fast YOLO Signature Detection</h1>
            <p>Lightning-fast signature detection using only YOLOv8</p>
            <span class="performance-badge">2-3 SECONDS</span>
        </div>

        <div class="main-content">
            <div class="info-box">
                <h4>🚀 How It Works</h4>
                <p><strong>YOLO Detection Only:</strong> No large language models, just pure object detection for maximum speed and reliability.</p>
                <p><strong>Processing Time:</strong> 2-3 seconds per image</p>
                <p><strong>Accuracy:</strong> YOLOv8's state-of-the-art object detection</p>
            </div>

            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div id="uploadContent">
                    <h3>📁 Upload Document Image</h3>
                    <p>Click to select or drag & drop</p>
                    <p style="color: #666; font-size: 0.9rem;">Supports: JPG, PNG, WebP</p>
                </div>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>🔍 Detecting signatures...</p>
                    <p style="font-size: 0.9em; color: #666;">This should take 2-3 seconds</p>
                </div>
            </div>

            <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="handleFileSelect(event)">

            <div class="results" id="results">
                <h2 style="text-align: center; margin-bottom: 30px; color: #667eea;">
                    📊 Detection Results
                </h2>

                <div class="image-container">
                    <img id="processedImage" class="processed-image" alt="Processed document">
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="signaturesCount">0</div>
                        <div class="stat-label">Signatures Found</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="processingTime">0.0s</div>
                        <div class="stat-label">Processing Time</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="regionsCount">0</div>
                        <div class="stat-label">Regions Detected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="accuracy">YOLOv8</div>
                        <div class="stat-label">Detection Method</div>
                    </div>
                </div>

                <div class="regions-list" id="regionsList">
                    <h3 style="margin-bottom: 15px; color: #667eea;">🔍 Detected Regions</h3>
                    <div id="regionsContent">
                        <p style="text-align: center; color: #666; padding: 20px;">
                            No signatures detected yet
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFile = null;

        // Drag and drop functionality
        const uploadArea = document.querySelector('.upload-area');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                handleFile(file);
            }
        }

        function handleFile(file) {
            currentFile = file;

            // Show file info
            const uploadContent = document.getElementById('uploadContent');
            uploadContent.innerHTML = `
                <h3>✅ File Selected</h3>
                <p><strong>${file.name}</strong></p>
                <p style="color: #666;">${(file.size / 1024 / 1024).toFixed(2)} MB</p>
                <button class="btn" onclick="processImage()">🚀 Detect Signatures</button>
            `;
        }

        async function processImage() {
            if (!currentFile) {
                alert('Please select a file first');
                return;
            }

            // Show loading
            const uploadArea = document.querySelector('.upload-area');
            const uploadContent = document.getElementById('uploadContent');
            const loading = document.getElementById('loading');

            uploadContent.style.display = 'none';
            loading.style.display = 'block';
            uploadArea.classList.add('uploading');

            try {
                const formData = new FormData();
                formData.append('file', currentFile);

                console.log('Sending file for processing...');

                const response = await fetch('/api/detect', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                console.log('Result:', result);

                if (result.success) {
                    displayResults(result);
                    uploadArea.classList.remove('uploading');
                    uploadArea.classList.add('success');
                } else {
                    alert('Error: ' + (result.error || 'Unknown error'));
                    uploadArea.classList.remove('uploading');
                }

            } catch (error) {
                console.error('Error:', error);
                alert('Processing failed: ' + error.message);
                uploadArea.classList.remove('uploading');
            } finally {
                loading.style.display = 'none';
            }
        }

        function displayResults(result) {
            // Show results
            const resultsDiv = document.getElementById('results');
            resultsDiv.classList.add('show');

            // Update image
            const img = document.getElementById('processedImage');
            img.src = result.image;

            // Update stats
            document.getElementById('signaturesCount').textContent = result.signatures_found;
            document.getElementById('processingTime').textContent = result.processing_time;
            document.getElementById('regionsCount').textContent = result.regions.length;

            // Update regions list
            const regionsContent = document.getElementById('regionsContent');

            if (result.regions.length > 0) {
                let html = '';
                result.regions.forEach(region => {
                    html += `
                        <div class="region-item">
                            <div><strong>Region ${region.id}</strong></div>
                            <div><strong>Confidence:</strong> ${(region.confidence * 100).toFixed(1)}%</div>
                            <div><strong>Area:</strong> ${region.area.toLocaleString()}</div>
                            <div><strong>Method:</strong> ${region.method}</div>
                        </div>
                    `;
                });
                regionsContent.innerHTML = html;
            } else {
                regionsContent.innerHTML = `
                    <p style="text-align: center; color: #666; padding: 20px;">
                        No signature regions detected in this image.<br>
                        Try uploading a different document with clearer signatures.
                    </p>
                `;
            }

            // Scroll to results
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🚀 YOLO Signature Detection UI loaded');
            console.log('⚡ Ready for super fast signature detection!');
        });
    </script>
</body>
</html>
        """

def main():
    """Main function to run the YOLO UI server"""
    print("⚡ Starting Super Fast YOLO Signature Detection UI")
    print("=" * 60)
    print("🚀 Features:")
    print("   • ⚡ 2-3 seconds processing time")
    print("   • 🎯 YOLOv8 object detection only")
    print("   • 🔍 No large language models")
    print("   • 📊 Detailed region information")
    print("   • 🎨 Modern, responsive UI")
    print("=" * 60)

    server = YOLOUIServer()

    if server.detector and server.detector.yolo_model:
        print("🌐 Starting server...")
        print("📋 UI available at: http://127.0.0.1:7862")
        print("💡 Upload any document and see instant signature detection!")
        print("=" * 60)

        uvicorn.run(server.app, host="127.0.0.1", port=7862, log_level="info")
    else:
        print("❌ Failed to initialize YOLO model")
        print("🔧 Make sure weights/yolov8s.pt exists")

if __name__ == "__main__":
    main()
