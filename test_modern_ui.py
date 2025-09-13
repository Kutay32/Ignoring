#!/usr/bin/env python3
"""
Quick test for Modern UI - No Gradio
"""

import sys
import os
import time
import webbrowser
from pathlib import Path

def test_modern_ui():
    """Test the modern UI quickly"""
    print("🧪 Testing Modern UI (No Gradio)")
    print("=" * 40)
    
    # Check if we can import required modules
    try:
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI not installed")
        return False
    
    try:
        import uvicorn
        print(f"✅ Uvicorn: {uvicorn.__version__}")
    except ImportError:
        print("❌ Uvicorn not installed")
        return False
    
    # Check if weights exist
    weights_path = "weights/yolov8s.pt"
    if os.path.exists(weights_path):
        print(f"✅ YOLO weights: {weights_path}")
    else:
        print(f"❌ YOLO weights not found: {weights_path}")
        return False
    
    # Test basic FastAPI app
    try:
        from modern_ui_backend import ModernSignatureAPI
        print("✅ Modern UI backend imported successfully")
        
        # Create API instance
        api = ModernSignatureAPI()
        print("✅ API instance created successfully")
        
        print("\n🚀 Starting test server...")
        print("🌐 Will be available at: http://127.0.0.1:7861")
        print("⏱️  Server will run for 10 seconds for testing")
        print("=" * 40)
        
        # Start server in background
        import threading
        import uvicorn
        
        def run_server():
            uvicorn.run(api.app, host="127.0.0.1", port=7861, log_level="error")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Try to open browser
        try:
            webbrowser.open("http://127.0.0.1:7861")
            print("🌐 Browser opened automatically")
        except:
            print("💡 Please open browser manually: http://127.0.0.1:7861")
        
        print("✅ Test server started successfully!")
        print("📋 Check your browser - you should see the modern UI")
        print("⏱️  Server will stop automatically in 10 seconds...")
        
        # Let it run for 10 seconds
        time.sleep(10)
        
        print("\n✅ Test completed successfully!")
        print("🎉 Modern UI is working - no 404 errors!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Modern UI: {e}")
        return False

if __name__ == "__main__":
    success = test_modern_ui()
    if success:
        print("\n🎯 Ready to use Modern UI!")
        print("Run: python launch_modern_ui.py")
    else:
        print("\n❌ Test failed. Check the errors above.")
