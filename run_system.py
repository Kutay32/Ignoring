#!/usr/bin/env python3
"""
Main launcher script for the Signature Extraction and Comparison System
"""

import sys
import os
import argparse
from gradio_ui import main as launch_ui
from test_system import run_comprehensive_test

def main():
    parser = argparse.ArgumentParser(description="Signature Extraction and Comparison System")
    parser.add_argument("--mode", choices=["ui", "test"], default="ui", 
                       help="Mode to run: 'ui' for web interface, 'test' for testing")
    parser.add_argument("--host", default="0.0.0.0", help="Host for web interface")
    parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
    parser.add_argument("--share", action="store_true", help="Create public link for web interface")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        print("🧪 Running comprehensive test suite...")
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    
    elif args.mode == "ui":
        print("🚀 Launching web interface...")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Share: {args.share}")
        print("\n📖 Usage Instructions:")
        print("   1. Load a model from the dropdown")
        print("   2. Upload a document image with signatures")
        print("   3. Set similarity threshold and user ID")
        print("   4. Click 'Process Image' to analyze")
        print("   5. Use 'Model Comparison' tab to compare all models")
        print("\n🌐 Web interface will open in your browser...")
        
        # Launch the Gradio interface
        launch_ui()

if __name__ == "__main__":
    main()