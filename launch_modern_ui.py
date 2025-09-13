#!/usr/bin/env python3
"""
Modern UI Launcher - No Gradio, No 404 Errors!
"""

import sys
import os
import subprocess
import webbrowser
import time
import signal

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n👋 Received interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'torch',
        'transformers',
        'ultralytics'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "fastapi", "uvicorn[standard]", "python-multipart"
            ])
            print("✅ FastAPI packages installed!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print("pip install fastapi uvicorn[standard] python-multipart")
            return False
    
    return True

def check_weights():
    """Check if YOLO weights exist"""
    weights_path = "weights/yolov8s.pt"
    if os.path.exists(weights_path):
        print(f"✅ YOLO weights found: {weights_path}")
        return True
    else:
        print(f"❌ YOLO weights not found: {weights_path}")
        print("Please download the weights file first.")
        return False

def launch_modern_ui():
    """Launch the modern UI"""
    print("🚀 Starting Modern Signature Detection UI...")
    print("=" * 60)
    print("✨ Features:")
    print("   • No Gradio dependencies")
    print("   • No 404 errors")
    print("   • Modern responsive design")
    print("   • Fast API backend")
    print("   • Kaggle T4x2 GPU optimized")
    print("   • 7B parameter model support")
    print("=" * 60)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Import and run the modern UI
        from modern_ui_backend import ModernSignatureAPI
        
        print("🌐 Starting server...")
        print("📋 UI will be available at: http://127.0.0.1:7860")
        print("💡 The browser should open automatically")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Create and run the API
        api = ModernSignatureAPI()
        api.run(host="127.0.0.1", port=7860)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure all dependencies are installed")
        print("2. Check if port 7860 is available")
        print("3. Try running as administrator")
        return False
    
    return True

def main():
    """Main function"""
    print("🎯 Modern Signature Detection System")
    print("🔧 FastAPI + Modern Web UI (Kaggle T4x2 Optimized)")
    print("🛠️  No Gradio - No 404 Errors!")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed")
        return
    
    # Check weights
    if not check_weights():
        print("❌ Weights check failed")
        return
    
    print("\n✅ All checks passed!")
    print()
    
    # Launch the UI
    success = launch_modern_ui()
    
    if success:
        print("\n✅ Modern UI launched successfully!")
    else:
        print("\n❌ Failed to launch Modern UI")

if __name__ == "__main__":
    main()
