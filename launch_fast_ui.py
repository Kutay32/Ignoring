#!/usr/bin/env python3
"""
Fast UI Launcher - Uses smaller model for quick testing
"""

import sys
import os
import signal

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n👋 Received interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def launch_fast_ui():
    """Launch the UI with a smaller, faster model"""
    print("🚀 Starting FAST Signature Detection UI")
    print("⚡ Uses smaller model for quick testing")
    print("=" * 50)

    # Modify the model to use a smaller one
    os.environ["QWEN_MODEL"] = "Qwen/Qwen2-VL-2B-Instruct"

    try:
        from modern_ui_backend import ModernSignatureAPI

        print("🌐 Starting server with fast model...")
        print("📋 UI will be available at: http://127.0.0.1:7860")
        print("💡 This should load much faster!")
        print("=" * 50)

        api = ModernSignatureAPI()

        # Override the model in the detector
        if hasattr(api, 'detector') and api.detector:
            api.detector.model_name = "Qwen/Qwen2-VL-2B-Instruct"
            print("🔄 Switched to smaller Qwen2-VL-2B model")

        api.run(host="127.0.0.1", port=7860)

    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Try the even simpler version:")
        print("python launch_yolo_only.py")

def create_yolo_only_version():
    """Create a version that only uses YOLO (no Qwen)"""
    yolo_only_code = '''#!/usr/bin/env python3
"""
YOLO-Only UI - No LLM, super fast
"""

import sys
import os
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from PIL import Image
import numpy as np
import time
from advanced_signature_detector import AdvancedSignatureDetector

class YOLOUi:
    def __init__(self):
        self.app = FastAPI(title="YOLO Signature Detection")
        self.detector = None
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def serve_ui():
            return HTMLResponse(content=self.get_html())

        @self.app.post("/api/process")
        async def process_image(file: UploadFile = File(...)):
            if not self.detector:
                self.detector = AdvancedSignatureDetector()
                self.detector._load_yolo_model()

            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))

            temp_path = f"/tmp/yolo_test_{int(time.time())}.png"
            image.save(temp_path)

            # Only YOLO detection - no Qwen
            regions = self.detector.detect_signature_regions_yolo(temp_path)

            # Convert to base64
            import base64
            from io import BytesIO
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            os.remove(temp_path)

            return {
                "success": True,
                "image": f"data:image/png;base64,{img_str}",
                "signatures_found": len(regions),
                "regions": [{"bbox": r["bbox"], "confidence": float(r["confidence"])} for r in regions],
                "processing_time": "2-3 seconds"
            }

    def get_html(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLO Signature Detection</title>
            <style>
                body { font-family: Arial; margin: 40px; background: #f0f0f0; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                .upload { border: 2px dashed #007bff; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; }
                .result { margin-top: 20px; padding: 20px; background: #e9ecef; border-radius: 10px; }
                img { max-width: 100%; border-radius: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 YOLO Signature Detection</h1>
                <p>⚡ Super fast - only YOLO, no LLM</p>

                <div class="upload" onclick="document.getElementById('file').click()">
                    <p>📁 Click to upload image</p>
                    <input type="file" id="file" accept="image/*" onchange="uploadFile()" style="display: none;">
                </div>

                <div id="result" class="result" style="display: none;">
                    <h3>Results:</h3>
                    <div id="content"></div>
                </div>
            </div>

            <script>
                async function uploadFile() {
                    const fileInput = document.getElementById('file');
                    const file = fileInput.files[0];
                    if (!file) return;

                    const formData = new FormData();
                    formData.append('file', file);

                    document.querySelector('.upload').innerHTML = '<p>🔄 Processing...</p>';

                    try {
                        const response = await fetch('/api/process', {
                            method: 'POST',
                            body: formData
                        });

                        const result = await response.json();

                        if (result.success) {
                            document.getElementById('result').style.display = 'block';
                            document.getElementById('content').innerHTML = `
                                <p><strong>Signatures Found:</strong> ${result.signatures_found}</p>
                                <p><strong>Processing Time:</strong> ${result.processing_time}</p>
                                <img src="${result.image}" alt="Processed image">
                                <p><small>YOLO detected ${result.signatures_found} potential signature regions</small></p>
                            `;
                        }
                    } catch (error) {
                        alert('Error: ' + error.message);
                    }

                    document.querySelector('.upload').innerHTML = '<p>📁 Click to upload image</p>';
                }
            </script>
        </body>
        </html>
        """

if __name__ == "__main__":
    ui = YOLOUi()
    uvicorn.run(ui.app, host="127.0.0.1", port=7861)
'''

    with open("launch_yolo_only.py", "w") as f:
        f.write(yolo_only_code)
    print("✅ Created launch_yolo_only.py")

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🎯 Fast UI Options:")
    print("1. Fast model (2B parameters)")
    print("2. YOLO-only (no LLM)")
    print("3. Exit")
    choice = input("Choose option (1-3): ").strip()

    if choice == "1":
        launch_fast_ui()
    elif choice == "2":
        create_yolo_only_version()
        print("🔥 Run: python launch_yolo_only.py")
    else:
        print("👋 Exiting")
