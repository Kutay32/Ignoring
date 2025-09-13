#!/usr/bin/env python3
"""
Modern Web UI Backend for Signature Detection
FastAPI + Modern Frontend (No Gradio dependencies)
"""

import sys
import os
import json
import base64
import io
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import numpy as np
import time
import torch

# Import our signature detection system
from advanced_signature_detector import AdvancedSignatureDetector

class ModernSignatureAPI:
    def __init__(self):
        self.app = FastAPI(
            title="Advanced Signature Detection System",
            description="Modern Web UI for Signature Detection using YOLOv8 + Qwen2.5-VL",
            version="2.0.0"
        )
        self.detector = None
        self.current_model = None
        self.setup_routes()
        self.setup_middleware()
    
    def setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def serve_ui():
            """Serve the main UI"""
            return HTMLResponse(content=self.get_ui_html(), status_code=200)
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": self.detector is not None
            }
        
        @self.app.post("/api/load_model")
        async def load_model(model_name: str = Form(...)):
            """Load the specified model"""
            try:
                self.detector = AdvancedSignatureDetector(
                    model_name=model_name,
                    yolo_weights_path="weights/yolov8s.pt",
                    use_quantization=True
                )
                self.detector.load_model()
                self.current_model = model_name
                
                return {
                    "success": True,
                    "message": f"Model {model_name} loaded successfully",
                    "device": str(self.detector.device)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
        
        @self.app.post("/api/process_image")
        async def process_image(
            file: UploadFile = File(...),
            user_id: Optional[str] = Form(None),
            similarity_threshold: float = Form(0.7),
            fast_mode: bool = Form(True)  # Enable fast mode by default
        ):
            """Process uploaded image for signature detection"""
            if self.detector is None:
                raise HTTPException(status_code=400, detail="Please load a model first")

            try:
                # Read and process image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))

                # Save temporarily
                temp_path = f"/tmp/temp_signature_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                image.save(temp_path)

                start_time = time.time()

                if fast_mode:
                    # HYBRID MODE: YOLO detection + targeted Qwen analysis on cropped regions
                    print("🎯 Using HYBRID MODE - YOLO detection + targeted Qwen analysis")

                    # Step 1: YOLO detects signature regions
                    regions = self.detector.detect_signature_regions_yolo(temp_path)
                    print(f"YOLO detected {len(regions)} potential signature regions")

                    # Step 2: For each region, crop and analyze with lightweight Qwen inference
                    signatures = []
                    for i, region in enumerate(regions):
                        try:
                            # Crop the signature region
                            x1, y1, x2, y2 = region['bbox']
                            cropped = image.crop((x1, y1, x2, y2))

                            # Quick Qwen analysis on cropped signature
                            analysis_result = self._quick_signature_analysis(cropped, region)

                            signature_data = {
                                "id": i + 1,
                                "bbox": region['bbox'],
                                "confidence": region['confidence'],
                                "method": region['method'],
                                "cropped_image": cropped,
                                "type": analysis_result.get("signature_type", "detected"),
                                "features": analysis_result.get("features", {}),
                                "qwen_analysis": analysis_result.get("analysis", "Quick analysis performed"),
                                "quality_score": analysis_result.get("quality_score", 0.5)
                            }
                            signatures.append(signature_data)
                            print(f"  ✓ Signature {i+1}: {analysis_result.get('signature_type', 'unknown')} (confidence: {region['confidence']:.2f})")

                        except Exception as e:
                            print(f"  ⚠️ Error analyzing signature {i+1}: {e}")
                            # Fallback to basic data
                            x1, y1, x2, y2 = region['bbox']
                            cropped = image.crop((x1, y1, x2, y2))

                            signature_data = {
                                "id": i + 1,
                                "bbox": region['bbox'],
                                "confidence": region['confidence'],
                                "method": region['method'],
                                "cropped_image": cropped,
                                "type": "detected",
                                "features": {"error": str(e)},
                                "qwen_analysis": "Analysis failed",
                                "quality_score": 0.0
                            }
                            signatures.append(signature_data)

                    processing_time = time.time() - start_time
                    print(f"Hybrid mode processing completed in {processing_time:.2f} seconds")
                    result = {
                        "success": True,
                        "signatures": signatures,
                        "processing_time": processing_time,
                        "regions_detected": len(regions),
                        "signatures_analyzed": len(signatures),
                        "mode": "hybrid",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # FULL MODE: Use complete VLM analysis (slow)
                    print("🧠 Using FULL MODE - Complete VLM analysis")
                    result = self.detector.process_document_advanced(temp_path, user_id or "anonymous")

                if not result["success"]:
                    raise HTTPException(status_code=500, detail=result["error"])

                # Convert cropped signatures to base64 for display
                cropped_signatures = []
                if "signatures" in result and result["signatures"]:
                    for i, sig in enumerate(result["signatures"]):
                        if "cropped_image" in sig and sig["cropped_image"] is not None:
                            # Convert PIL image to base64
                            buffered = io.BytesIO()
                            sig["cropped_image"].save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            cropped_signatures.append({
                                "id": i + 1,
                                "image": f"data:image/png;base64,{img_str}",
                                "confidence": sig.get("confidence", sig.get("detection_confidence", 0)),
                                "type": sig.get("type", "detected"),
                                "location": f"{sig.get('bbox', [0,0,0,0])}",
                                "method": sig.get("method", sig.get("detection_method", "unknown"))
                            })

                # Convert original image to base64 for display
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                original_img_str = base64.b64encode(buffered.getvalue()).decode()

                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                return {
                    "success": True,
                    "result": result,
                    "original_image": f"data:image/png;base64,{original_img_str}",
                    "cropped_signatures": cropped_signatures,
                    "processing_time": result.get("processing_time", 0),
                    "signatures_found": len(result.get("signatures", [])),
                    "mode": "fast" if fast_mode else "full",
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    def _quick_signature_analysis(self, cropped_signature: Image.Image, region_info: dict) -> dict:
        """
        Perform quick, targeted Qwen analysis on a cropped signature region.
        Much faster than full document analysis since it focuses on individual signatures.
        """
        try:
            # Ensure model is loaded
            if self.detector.model is None or self.detector.processor is None:
                print("Loading Qwen model for signature analysis...")
                self.detector.load_model()

            # Resize cropped signature for efficient processing
            max_size = 512  # Smaller than full document analysis
            if max(cropped_signature.size) > max_size:
                ratio = max_size / max(cropped_signature.size)
                new_size = (int(cropped_signature.size[0] * ratio), int(cropped_signature.size[1] * ratio))
                cropped_signature = cropped_signature.resize(new_size, Image.Resampling.LANCZOS)

            # Quick analysis prompt focused on signature characteristics
            quick_prompt = """
            Analyze this signature image quickly and provide:
            1. SIGNATURE TYPE: [cursive/print/mixed/artistic/symbolic]
            2. QUALITY SCORE: [high/medium/low] based on clarity and completeness
            3. KEY FEATURES: Brief description of distinctive characteristics
            4. CONFIDENCE: How confident you are this is a valid signature

            Keep response brief and focused.
            """

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": cropped_signature},
                        {"type": "text", "text": quick_prompt}
                    ]
                }
            ]

            text = self.detector.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.detector.processor(
                text=[text],
                images=[cropped_signature],
                videos=None,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.detector.device) for k, v in inputs.items()}

            # Generate with shorter response (faster)
            with torch.no_grad():
                generated_ids = self.detector.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Much shorter than full analysis
                    do_sample=False,
                    temperature=0.1
                )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], generated_ids)
            ]

            analysis = self.detector.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Parse the quick analysis
            signature_type = "unknown"
            quality_score = 0.5
            confidence = region_info.get('confidence', 0.5)

            # Simple parsing of the response
            analysis_lower = analysis.lower()
            if "cursive" in analysis_lower:
                signature_type = "cursive"
                quality_score = 0.8
            elif "print" in analysis_lower:
                signature_type = "print"
                quality_score = 0.7
            elif "mixed" in analysis_lower:
                signature_type = "mixed"
                quality_score = 0.75
            elif "artistic" in analysis_lower:
                signature_type = "artistic"
                quality_score = 0.85
            elif "symbolic" in analysis_lower:
                signature_type = "symbolic"
                quality_score = 0.6

            # Adjust quality based on response
            if "high" in analysis_lower:
                quality_score = min(quality_score + 0.2, 1.0)
            elif "low" in analysis_lower:
                quality_score = max(quality_score - 0.2, 0.1)

            return {
                "signature_type": signature_type,
                "quality_score": quality_score,
                "analysis": analysis[:200] + "..." if len(analysis) > 200 else analysis,  # Truncate long responses
                "confidence": confidence,
                "features": {
                    "bbox": region_info.get('bbox', []),
                    "area": region_info.get('area', 0),
                    "aspect_ratio": region_info.get('aspect_ratio', 1.0),
                    "method": region_info.get('method', 'yolo'),
                    "yolo_class": region_info.get('yolo_class', -1)
                }
            }

        except Exception as e:
            print(f"Error in quick signature analysis: {e}")
            # Return fallback data
            return {
                "signature_type": "detected",
                "quality_score": 0.5,
                "analysis": f"Basic detection completed (analysis failed: {str(e)})",
                "confidence": region_info.get('confidence', 0.5),
                "features": {
                    "bbox": region_info.get('bbox', []),
                    "area": region_info.get('area', 0),
                    "aspect_ratio": region_info.get('aspect_ratio', 1.0),
                    "error": str(e)
                }
            }
        
        @self.app.get("/api/database_stats")
        async def get_database_stats():
            """Get database statistics"""
            if self.detector is None:
                raise HTTPException(status_code=400, detail="Please load a model first")
            
            try:
                stats = self.detector.get_database_stats()
                return {
                    "success": True,
                    "stats": stats,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")
        
        @self.app.post("/api/compare_signatures")
        async def compare_signatures(
            sig1_id: int = Form(...),
            sig2_id: int = Form(...)
        ):
            """Compare two signatures"""
            if self.detector is None:
                raise HTTPException(status_code=400, detail="Please load a model first")
            
            try:
                comparison = self.detector.compare_signatures(sig1_id, sig2_id)
                return {
                    "success": True,
                    "comparison": comparison,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error comparing signatures: {str(e)}")
    
    def get_ui_html(self) -> str:
        """Generate the modern UI HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Signature Detection System</title>
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
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .main-content {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 30px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .tab {
            padding: 15px 25px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1.1rem;
            font-weight: 600;
            color: #666;
            transition: all 0.3s ease;
            border-bottom: 3px solid transparent;
        }
        
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        
        .tab:hover {
            color: #667eea;
            background: #f8f9ff;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .form-group input,
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        .form-group input:focus,
        .form-group select:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-secondary {
            background: #6c757d;
        }
        
        .btn-success {
            background: #28a745;
        }
        
        .btn-danger {
            background: #dc3545;
        }
        
        .status {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: 600;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        
        .file-upload {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .file-upload:hover {
            background: #f0f2ff;
            border-color: #5a6fd8;
        }
        
        .file-upload.dragover {
            background: #e8f0ff;
            border-color: #4c63d2;
        }
        
        .results {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .card h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .hidden {
            display: none;
        }
        
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
        }

        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }

        .badge.fast {
            background: #28a745;
            color: white;
        }

        .badge.full {
            background: #dc3545;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Advanced Signature Detection System</h1>
            <p>Powered by YOLOv8 + Qwen2.5-VL | Modern Web Interface</p>
        </div>
        
        <div class="main-content">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('detection')">🔍 Signature Detection</button>
                <button class="tab" onclick="switchTab('comparison')">🔄 Signature Comparison</button>
                <button class="tab" onclick="switchTab('database')">📊 Database</button>
                <button class="tab" onclick="switchTab('settings')">⚙️ Settings</button>
            </div>
            
            <!-- Detection Tab -->
            <div id="detection" class="tab-content active">
                <div class="form-group">
                    <label for="modelSelect">Select AI Model:</label>
                    <select id="modelSelect">
                        <option value="Qwen/Qwen2.5-VL-7B-Instruct">Qwen2.5-VL-7B-Instruct (Best Performance - T4x2 Optimized)</option>
                        <option value="Qwen/Qwen2.5-VL-3B-Instruct">Qwen2.5-VL-3B-Instruct (Balanced)</option>
                        <option value="Qwen/Qwen2-VL-2B-Instruct">Qwen2-VL-2B-Instruct (Fastest)</option>
                    </select>
                    <button class="btn" onclick="loadModel()">Load Model</button>
                </div>
                
                <div id="modelStatus" class="status hidden"></div>
                
                <div class="form-group">
                    <label>Upload Document Image:</label>
                    <div class="file-upload" onclick="document.getElementById('imageInput').click()">
                        <p>📁 Click to upload or drag & drop image</p>
                        <p style="color: #666; font-size: 0.9rem;">Supports: JPG, PNG, WebP</p>
                    </div>
                    <input type="file" id="imageInput" accept="image/*" onchange="handleFileSelect(event)">
                </div>
                
                <div class="form-group">
                    <label for="userId">User ID (Optional):</label>
                    <input type="text" id="userId" placeholder="Enter user ID for storage">
                </div>
                
                <div class="form-group">
                    <label for="threshold">Similarity Threshold:</label>
                    <input type="range" id="threshold" min="0.1" max="1.0" step="0.1" value="0.7">
                    <span id="thresholdValue">0.7</span>
                </div>

                <div class="form-group">
                    <label>Processing Mode:</label>
                    <div style="display: flex; gap: 20px; align-items: center;">
                        <label style="display: flex; align-items: center; gap: 5px;">
                            <input type="radio" name="processingMode" value="fast" checked>
                            <span>🎯 Hybrid Mode</span>
                            <small style="color: #666;">(YOLO + targeted Qwen, ~30-90 seconds)</small>
                        </label>
                        <label style="display: flex; align-items: center; gap: 5px;">
                            <input type="radio" name="processingMode" value="full">
                            <span>🧠 Full Analysis</span>
                            <small style="color: #666;">(Complete VLM analysis, ~5-30 minutes)</small>
                        </label>
                    </div>
                    <div style="margin-top: 8px; padding: 8px; background: #f0f8ff; border-radius: 6px; border-left: 3px solid #667eea;">
                        <small style="color: #444;">
                            <strong>🎯 Hybrid Mode:</strong> YOLO detects signature regions → Qwen analyzes each cropped signature individually for better accuracy with reasonable speed.
                        </small>
                    </div>
                </div>

                <button class="btn btn-success" onclick="processImage()">Process Image</button>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing image...</p>
                </div>
                
                <!-- Results Section with Image Display -->
                <div id="results" class="results hidden">
                    <h3>🔍 Processing Results <span id="processingModeBadge" class="badge"></span></h3>
                    
                    <!-- Original Image and Cropped Signatures -->
                    <div class="grid" style="grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
                        <!-- Left: Original Image -->
                        <div class="card">
                            <h3>📄 Original Document</h3>
                            <div id="originalImageContainer" style="text-align: center;">
                                <img id="originalImage" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" />
                            </div>
                        </div>
                        
                        <!-- Right: Cropped Signatures -->
                        <div class="card">
                            <h3>✍️ Detected Signatures</h3>
                            <div id="signaturesContainer">
                                <p style="text-align: center; color: #666; padding: 20px;">No signatures detected yet</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Processing Summary -->
                    <div class="card" style="margin-top: 20px;">
                        <h3>📊 Processing Summary</h3>
                        <div id="summaryContent"></div>
                    </div>
                    
                    <!-- Detailed Analysis -->
                    <div class="card" style="margin-top: 20px;">
                        <h3>🔍 Detailed Analysis</h3>
                        <div id="analysisContent"></div>
                    </div>
                </div>
            </div>
            
            <!-- Comparison Tab -->
            <div id="comparison" class="tab-content">
                <div class="grid">
                    <div class="card">
                        <h3>Signature 1</h3>
                        <div class="form-group">
                            <label for="sig1Id">Signature ID:</label>
                            <input type="number" id="sig1Id" value="1" min="1">
                        </div>
                    </div>
                    <div class="card">
                        <h3>Signature 2</h3>
                        <div class="form-group">
                            <label for="sig2Id">Signature ID:</label>
                            <input type="number" id="sig2Id" value="2" min="1">
                        </div>
                    </div>
                </div>
                <button class="btn" onclick="compareSignatures()">Compare Signatures</button>
                <div id="comparisonResults" class="results hidden">
                    <h3>Comparison Results</h3>
                    <div id="comparisonContent"></div>
                </div>
            </div>
            
            <!-- Database Tab -->
            <div id="database" class="tab-content">
                <button class="btn" onclick="getDatabaseStats()">Get Database Statistics</button>
                <div id="databaseResults" class="results hidden">
                    <h3>Database Statistics</h3>
                    <div id="databaseContent"></div>
                </div>
            </div>
            
            <!-- Settings Tab -->
            <div id="settings" class="tab-content">
                <div class="card">
                    <h3>System Information</h3>
                    <div id="systemInfo">
                        <p><strong>Status:</strong> <span id="systemStatus">Checking...</span></p>
                        <p><strong>Model Loaded:</strong> <span id="modelLoaded">No</span></p>
                        <p><strong>Last Update:</strong> <span id="lastUpdate">-</span></p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Global variables
        let currentModel = null;
        let isProcessing = false;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM loaded, initializing...');
            updateSystemStatus();
            const threshold = document.getElementById('threshold');
            const thresholdValue = document.getElementById('thresholdValue');

            if (threshold && thresholdValue) {
                threshold.addEventListener('input', function() {
                    thresholdValue.textContent = this.value;
                });
            } else {
                console.error('Threshold elements not found');
            }

            // Verify file input exists
            const fileInput = document.getElementById('imageInput');
            if (!fileInput) {
                console.error('File input element not found!');
            } else {
                console.log('File input element found');
            }
        });
        
        // Tab switching
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        // Model loading
        async function loadModel() {
            const modelSelect = document.getElementById('modelSelect');
            const statusDiv = document.getElementById('modelStatus');
            
            statusDiv.className = 'status info';
            statusDiv.textContent = 'Loading model...';
            statusDiv.classList.remove('hidden');
            
            try {
                const formData = new FormData();
                formData.append('model_name', modelSelect.value);
                
                const response = await fetch('/api/load_model', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    statusDiv.className = 'status success';
                    statusDiv.textContent = `✅ ${result.message} (Device: ${result.device})`;
                    currentModel = modelSelect.value;
                    updateSystemStatus();
                } else {
                    statusDiv.className = 'status error';
                    statusDiv.textContent = `❌ ${result.message}`;
                }
            } catch (error) {
                statusDiv.className = 'status error';
                statusDiv.textContent = `❌ Error: ${error.message}`;
            }
        }
        
        // File handling
        function handleFileSelect(event) {
            console.log('File selection changed');
            const fileInput = event.target;
            const file = fileInput.files ? fileInput.files[0] : null;

            if (file) {
                console.log('File selected:', file.name, 'Size:', file.size);
                const uploadDiv = document.querySelector('.file-upload');
                uploadDiv.innerHTML = `
                    <p>✅ Selected: ${file.name}</p>
                    <p style="color: #666; font-size: 0.9rem;">Size: ${(file.size / 1024 / 1024).toFixed(2)} MB</p>
                `;
            } else {
                console.log('No file selected');
            }
        }
        
        // Image processing
        async function processImage() {
            console.log('processImage called');

            if (!currentModel) {
                alert('Please load a model first!');
                return;
            }

            const fileInput = document.getElementById('imageInput');
            const userIdElement = document.getElementById('userId');
            const thresholdElement = document.getElementById('threshold');
            const processingMode = document.querySelector('input[name="processingMode"]:checked');

            console.log('File input element:', fileInput);

            // Check if all elements exist
            if (!fileInput) {
                alert('File input element not found. Please refresh the page.');
                console.error('fileInput element not found');
                return;
            }

            if (!userIdElement || !thresholdElement || !processingMode) {
                alert('Form elements not found. Please refresh the page.');
                console.error('userId, threshold, or processingMode element not found');
                return;
            }

            const userId = userIdElement.value || 'anonymous';
            const threshold = thresholdElement.value;
            const fastMode = processingMode.value === 'fast';

            console.log('Files in input:', fileInput.files);
            console.log('Processing mode:', fastMode ? 'FAST' : 'FULL');

            if (!fileInput.files || fileInput.files.length === 0) {
                alert('Please select an image first!');
                return;
            }

            if (isProcessing) {
                console.log('Already processing, ignoring request');
                return;
            }

            // Show warning for full mode
            if (!fastMode) {
                const confirmed = confirm('⚠️ Full Analysis Mode will take 5-30 minutes and use significant CPU/GPU resources.\\n\\nAre you sure you want to continue?');
                if (!confirmed) return;
            }

            isProcessing = true;
            console.log('Starting image processing...');

            const loadingDiv = document.getElementById('loading');
            const resultsDiv = document.getElementById('results');

            if (loadingDiv) {
                const modeText = fastMode ? 'Hybrid Mode (YOLO + Qwen)' : 'Full Analysis';
                const timeText = fastMode ? '~30-90 seconds' : '~5-30 minutes';
                loadingDiv.innerHTML = `
                    <div class="spinner"></div>
                    <p>Processing image with ${modeText}...</p>
                    <p style="font-size: 0.9em; color: #666;">Estimated time: ${timeText}</p>
                `;
                loadingDiv.style.display = 'block';
            }
            if (resultsDiv) resultsDiv.classList.add('hidden');

            try {
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('user_id', userId);
                formData.append('similarity_threshold', threshold);
                formData.append('fast_mode', fastMode.toString());

                console.log('Sending request to /api/process_image');

                const response = await fetch('/api/process_image', {
                    method: 'POST',
                    body: formData
                });

                console.log('Response received:', response.status);

                const result = await response.json();
                console.log('Result:', result);

                if (result.success) {
                    displayResults(result);
                } else {
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                console.error('Error in processImage:', error);
                alert(`Error: ${error.message}`);
            } finally {
                if (loadingDiv) loadingDiv.style.display = 'none';
                isProcessing = false;
                console.log('Image processing finished');
            }
        }
        
        // Display results
        function displayResults(apiResult) {
            const resultsDiv = document.getElementById('results');
            const originalImage = document.getElementById('originalImage');
            const signaturesContainer = document.getElementById('signaturesContainer');
            const summaryContent = document.getElementById('summaryContent');
            const analysisContent = document.getElementById('analysisContent');
            const modeBadge = document.getElementById('processingModeBadge');

            // Show processing mode badge
            if (modeBadge) {
                const mode = apiResult.mode || 'unknown';
                modeBadge.textContent = mode.toUpperCase();
                modeBadge.className = `badge ${mode}`;
            }

            // Show original image
            if (apiResult.original_image) {
                originalImage.src = apiResult.original_image;
                originalImage.style.display = 'block';
            }
            
            // Display cropped signatures
            if (apiResult.cropped_signatures && apiResult.cropped_signatures.length > 0) {
                let signaturesHtml = '';
                apiResult.cropped_signatures.forEach((sig, index) => {
                    signaturesHtml += `
                        <div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #667eea;">
                            <h4 style="color: #667eea; margin-bottom: 10px;">Signature ${sig.id}</h4>
                            <div style="text-align: center; margin: 10px 0;">
                                <img src="${sig.image}" style="max-width: 200px; max-height: 150px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" />
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                                <div><strong>Type:</strong> ${sig.type || 'Unknown'}</div>
                                <div><strong>YOLO Confidence:</strong> ${(sig.confidence * 100).toFixed(1)}%</div>
                                <div><strong>Quality Score:</strong> ${sig.quality_score ? (sig.quality_score * 100).toFixed(1) + '%' : 'N/A'}</div>
                                <div><strong>Method:</strong> ${sig.method || 'Unknown'}</div>
                            </div>
                            ${sig.qwen_analysis ? `<div style="margin-top: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em;"><strong>Qwen Analysis:</strong> ${sig.qwen_analysis}</div>` : ''}
                        </div>
                    `;
                });
                signaturesContainer.innerHTML = signaturesHtml;
            } else {
                signaturesContainer.innerHTML = '<p style="text-align: center; color: #666; padding: 20px;">No signatures detected in this image</p>';
            }
            
            // Display summary
            const result = apiResult.result;
            summaryContent.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div style="padding: 10px; background: #e8f5e8; border-radius: 8px;">
                        <strong>Signatures Found:</strong><br>
                        <span style="font-size: 1.5em; color: #28a745;">${apiResult.signatures_found}</span>
                    </div>
                    <div style="padding: 10px; background: #e3f2fd; border-radius: 8px;">
                        <strong>Processing Time:</strong><br>
                        <span style="font-size: 1.5em; color: #2196f3;">${apiResult.processing_time.toFixed(2)}s</span>
                    </div>
                    <div style="padding: 10px; background: #fff3e0; border-radius: 8px;">
                        <strong>Success Rate:</strong><br>
                        <span style="font-size: 1.5em; color: #ff9800;">${result.success ? '100%' : '0%'}</span>
                    </div>
                    <div style="padding: 10px; background: #f3e5f5; border-radius: 8px;">
                        <strong>Document Type:</strong><br>
                        <span style="font-size: 1.2em; color: #9c27b0;">${result.document_type || 'Unknown'}</span>
                    </div>
                </div>
            `;
            
            // Display detailed analysis
            analysisContent.innerHTML = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <h4 style="color: #667eea; margin-bottom: 10px;">📋 Processing Details</h4>
                        <ul style="list-style: none; padding: 0;">
                            <li style="padding: 5px 0; border-bottom: 1px solid #eee;"><strong>Quality Score:</strong> ${result.quality_score || 'N/A'}</li>
                            <li style="padding: 5px 0; border-bottom: 1px solid #eee;"><strong>Confidence:</strong> ${result.confidence || 'N/A'}</li>
                            <li style="padding: 5px 0; border-bottom: 1px solid #eee;"><strong>Processing Method:</strong> YOLOv8 + Qwen2.5-VL</li>
                            <li style="padding: 5px 0; border-bottom: 1px solid #eee;"><strong>Timestamp:</strong> ${new Date(apiResult.timestamp).toLocaleString()}</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="color: #667eea; margin-bottom: 10px;">🔍 Detection Analysis</h4>
                        <ul style="list-style: none; padding: 0;">
                            <li style="padding: 5px 0; border-bottom: 1px solid #eee;"><strong>Model Status:</strong> ${result.success ? '✅ Active' : '❌ Failed'}</li>
                            <li style="padding: 5px 0; border-bottom: 1px solid #eee;"><strong>Detection Method:</strong> Advanced AI Analysis</li>
                            <li style="padding: 5px 0; border-bottom: 1px solid #eee;"><strong>Image Resolution:</strong> ${result.image_resolution || 'Unknown'}</li>
                            <li style="padding: 5px 0; border-bottom: 1px solid #eee;"><strong>Storage Status:</strong> ${result.stored ? '✅ Saved' : '❌ Not Saved'}</li>
                        </ul>
                    </div>
                </div>
            `;
            
            resultsDiv.classList.remove('hidden');
        }
        
        // Compare signatures
        async function compareSignatures() {
            if (!currentModel) {
                alert('Please load a model first!');
                return;
            }
            
            const sig1Id = document.getElementById('sig1Id').value;
            const sig2Id = document.getElementById('sig2Id').value;
            
            try {
                const formData = new FormData();
                formData.append('sig1_id', sig1Id);
                formData.append('sig2_id', sig2Id);
                
                const response = await fetch('/api/compare_signatures', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayComparisonResults(result.comparison);
                } else {
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }
        
        // Display comparison results
        function displayComparisonResults(comparison) {
            const resultsDiv = document.getElementById('comparisonResults');
            const contentDiv = document.getElementById('comparisonContent');
            
            let html = `
                <div class="card">
                    <h3>🔄 Comparison Results</h3>
                    <p><strong>Similarity Score:</strong> ${comparison.similarity_score || 'N/A'}</p>
                    <p><strong>Match Status:</strong> ${comparison.match_status || 'Unknown'}</p>
                    <p><strong>Confidence:</strong> ${comparison.confidence || 'N/A'}</p>
                </div>
            `;
            
            contentDiv.innerHTML = html;
            resultsDiv.classList.remove('hidden');
        }
        
        // Get database stats
        async function getDatabaseStats() {
            if (!currentModel) {
                alert('Please load a model first!');
                return;
            }
            
            try {
                const response = await fetch('/api/database_stats');
                const result = await response.json();
                
                if (result.success) {
                    displayDatabaseStats(result.stats);
                } else {
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }
        
        // Display database stats
        function displayDatabaseStats(stats) {
            const resultsDiv = document.getElementById('databaseResults');
            const contentDiv = document.getElementById('databaseContent');
            
            let html = `
                <div class="card">
                    <h3>📊 Database Statistics</h3>
                    <p><strong>Total Signatures:</strong> ${stats.total_signatures || 0}</p>
                    <p><strong>Database Size:</strong> ${stats.database_size || 'N/A'}</p>
                    <p><strong>Last Updated:</strong> ${stats.last_updated || 'N/A'}</p>
                </div>
            `;
            
            contentDiv.innerHTML = html;
            resultsDiv.classList.remove('hidden');
        }
        
        // Update system status
        async function updateSystemStatus() {
            try {
                const response = await fetch('/api/health');
                const result = await response.json();
                
                document.getElementById('systemStatus').textContent = result.status;
                document.getElementById('modelLoaded').textContent = result.model_loaded ? 'Yes' : 'No';
                document.getElementById('lastUpdate').textContent = new Date(result.timestamp).toLocaleString();
            } catch (error) {
                document.getElementById('systemStatus').textContent = 'Error';
                document.getElementById('modelLoaded').textContent = 'Unknown';
            }
        }
        
        // Update system status every 30 seconds
        setInterval(updateSystemStatus, 30000);
    </script>
</body>
</html>
        """
    
    def run(self, host="127.0.0.1", port=7860):
        """Run the FastAPI server"""
        print("🚀 Starting Modern Signature Detection API...")
        print(f"🌐 Server will be available at: http://{host}:{port}")
        print("💡 No external dependencies - all resources are local!")
        print("=" * 60)
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )

def main():
    """Main function"""
    print("🎯 Modern Signature Detection System")
    print("🔧 FastAPI + Modern Web UI (No Gradio)")
    print("=" * 60)
    
    # Check if weights exist
    if not os.path.exists("weights/yolov8s.pt"):
        print("❌ YOLO weights not found at weights/yolov8s.pt")
        print("Please ensure the weights file exists before running.")
        return
    
    # Create and run the API
    api = ModernSignatureAPI()
    api.run()

if __name__ == "__main__":
    main()
