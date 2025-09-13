#!/usr/bin/env python3
"""
Script to run the Advanced Signature Detection UI with YOLOv8 + Qwen integration
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    """Launch the advanced signature detection UI"""
    print("🚀 Starting Advanced Signature Detection UI...")
    print("📋 Features:")
    print("  • YOLOv8 object detection for signature regions")
    print("  • Qwen VLM for detailed signature analysis")
    print("  • OpenCV fallback detection")
    print("  • Advanced similarity comparison")
    print("  • Database storage and management")
    print()

    try:
        from advanced_gradio_ui import main as ui_main
        ui_main()
    except ImportError as e:
        print(f"❌ Error importing UI: {e}")
        print("Make sure all required packages are installed:")
        print("pip install gradio ultralytics torch transformers accelerate")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 UI stopped by user")
    except Exception as e:
        print(f"❌ Error running UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
