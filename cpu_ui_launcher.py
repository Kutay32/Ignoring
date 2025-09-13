#!/usr/bin/env python3
"""
CPU-only UI launcher for Signature Detection (if GPU memory is insufficient)
"""

import sys
import os
import signal
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n👋 Received interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def check_cpu_requirements():
    """Check if system has enough CPU memory"""
    print("🔍 Checking CPU memory requirements...")

    import psutil
    memory = psutil.virtual_memory()

    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)

    print(".1f")    
    print(f"Available RAM: {available_gb:.1f}GB")
    # Qwen2.5-VL-7B typically needs ~16GB+ RAM for CPU inference
    if available_gb < 16:
        print(f"⚠️  Warning: Only {available_gb:.1f}GB RAM available")
        print("   CPU inference may be very slow or fail with large models")
        print("   Consider using the GPU version or a smaller model")
        return False

    print("✅ Sufficient CPU memory available")
    return True

def launch_cpu_ui():
    """Launch UI with CPU-only settings"""
    print("🚀 Starting CPU-only Signature Detection UI...")
    print("⚠️  Note: CPU inference will be slower than GPU")
    print("=" * 60)

    try:
        # Force CPU usage by setting environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Import and modify the detector to use CPU
        from advanced_gradio_ui import AdvancedSignatureUI

        print("📦 Initializing CPU-only UI...")
        ui = AdvancedSignatureUI()

        print("🎨 Creating interface...")
        interface = ui.create_interface()

        print("🌐 Launching web interface (CPU mode)...")
        print("📋 UI will be available at: http://localhost:7861")
        print("🐌 CPU processing will be slower - please be patient!")
        print("=" * 60)

        interface.launch(
            server_name="0.0.0.0",
            server_port=7861,  # Different port to avoid conflicts
            share=False,
            show_error=True,
            quiet=False
        )

    except KeyboardInterrupt:
        print("\n👋 UI stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching CPU UI: {e}")
        print("📄 Full error traceback:")
        traceback.print_exc()
        return False

    return True

def main():
    """Main CPU launcher function"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🐌 CPU-Only Advanced Signature Detection System")
    print("🔧 Powered by YOLOv8 + Qwen2.5-VL-7B (CPU Mode)")
    print("=" * 60)

    # Check CPU requirements
    if not check_cpu_requirements():
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Exiting. Try using GPU version or free up more RAM.")
            sys.exit(1)

    print()
    success = launch_cpu_ui()

    if success:
        print("\n✅ CPU UI launched successfully!")
    else:
        print("\n❌ CPU UI failed to launch.")
        sys.exit(1)

if __name__ == "__main__":
    main()
