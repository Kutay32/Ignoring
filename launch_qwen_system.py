#!/usr/bin/env python3
"""
Simple launcher for the Qwen signature system
"""

import sys
import os

def main():
    """Launch the Qwen signature system"""
    print("🚀 Launching Qwen Signature System")
    print("=" * 50)
    
    try:
        # Test basic functionality first
        print("Testing system components...")
        
        # Test signature extractor
        from signature_extractor import SignatureExtractor
        print("✅ SignatureExtractor imported")
        
        # Test model loading
        print("Loading Qwen model (this may take a few minutes)...")
        extractor = SignatureExtractor()
        extractor.load_model()
        print("✅ Qwen model loaded successfully!")
        
        # Test web interface
        print("Starting web interface...")
        from gradio_ui import SignatureComparisonUI
        
        ui = SignatureComparisonUI()
        interface = ui.create_interface()
        
        print("\n🌐 Web interface starting...")
        print("   - URL: http://localhost:7860")
        print("   - Press Ctrl+C to stop")
        print("\n📖 Usage:")
        print("   1. Select a model from the dropdown")
        print("   2. Upload a document image with signatures")
        print("   3. Set similarity threshold and user ID")
        print("   4. Click 'Process Image' to analyze")
        
        interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
        
    except KeyboardInterrupt:
        print("\n👋 System stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check available memory (Qwen models need ~4GB+ RAM)")
        print("   3. Try the smaller 2B model if 7B+ models fail")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)