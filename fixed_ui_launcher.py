#!/usr/bin/env python3
"""
Fixed UI launcher for Signature Detection - addresses 404 errors
"""

import sys
import os
import time
import signal
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n👋 Received interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def check_system_requirements():
    """Check if system meets basic requirements"""
    print("🔍 Checking system requirements...")

    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check if CUDA is available
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA available: Yes (GPU: {torch.cuda.get_device_name(0)})")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        else:
            print("CUDA available: No (using CPU)")
    except ImportError:
        print("❌ PyTorch not installed!")
        return False

    # Check if required files exist
    yolo_path = "weights/yolov8s.pt"
    if os.path.exists(yolo_path):
        print(f"YOLO weights: ✅ Found at {yolo_path}")
    else:
        print(f"❌ YOLO weights not found at {yolo_path}")
        return False

    print("✅ System requirements check passed!")
    return True

def launch_ui_fixed():
    """Launch the UI with 404 error fixes"""
    print("🚀 Starting Advanced Signature Detection UI (Fixed Version)...")
    print("=" * 60)

    try:
        from advanced_gradio_ui import AdvancedSignatureUI

        print("📦 Initializing UI...")
        ui = AdvancedSignatureUI()

        print("🎨 Creating interface...")
        interface = ui.create_interface()

        print("🌐 Launching web interface...")
        print("📋 UI will be available at: http://127.0.0.1:7860")
        print("💡 Keep this terminal window open while using the UI")
        print("=" * 60)

        # Launch with 404 error fixes
        print("🚀 Launching interface with 404 error fixes...")
        print("🌐 Server configuration:")
        print(f"   - Host: 127.0.0.1 (localhost only)")
        print(f"   - Port: 7860")
        print(f"   - Share: False (no external resources)")
        print(f"   - Debug: False (minimal external loading)")
        print()
        print("💡 If the browser doesn't open automatically:")
        print("   Open your browser and go to: http://127.0.0.1:7860")
        print("   Or try: http://localhost:7860")
        print()
        print("🔧 404 Error Fixes Applied:")
        print("   - Disabled external sharing")
        print("   - Using localhost only (127.0.0.1)")
        print("   - Disabled debug mode")
        print("   - Minimized external resource loading")
        print()

        # Launch with minimal external dependencies
        interface.launch(
            server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
            server_port=7860,
            share=False,  # Disable sharing to avoid external issues
            show_error=True,
            debug=False,  # Disable debug to reduce external resource loading
            quiet=False,
            inbrowser=True  # Open browser automatically
        )

    except KeyboardInterrupt:
        print("\n👋 UI stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching UI: {e}")
        print("📄 Full error traceback:")
        traceback.print_exc()

        print("\n🔧 Troubleshooting suggestions:")
        print("1. Check if all required packages are installed:")
        print("   pip install torch torchvision transformers accelerate gradio ultralytics")
        print("2. Install bitsandbytes for memory efficiency:")
        print("   pip install bitsandbytes")
        print("3. Ensure you have enough RAM/VRAM (at least 8GB recommended)")
        print("4. Try restarting your computer")
        print("5. Check your internet connection for model downloads")
        print("6. Try running the diagnostic script: python diagnose_gradio_errors.py")

        return False

    return True

def main():
    """Main function with comprehensive error handling"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🎯 Advanced Signature Detection System (Fixed Version)")
    print("🔧 Powered by YOLOv8 + Qwen2.5-VL-7B")
    print("🛠️  Includes 404 error fixes")
    print("=" * 60)

    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please fix the issues above.")
        sys.exit(1)

    print()
    success = launch_ui_fixed()

    if success:
        print("\n✅ UI launched successfully!")
    else:
        print("\n❌ UI failed to launch. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
