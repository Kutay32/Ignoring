#!/usr/bin/env python3
"""
Debug UI test to identify why the interface appears blank
"""

import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_ui_creation():
    """Test if the UI can be created without launching"""
    print("🔍 Testing UI creation...")
    print("=" * 50)

    try:
        print("📦 Importing AdvancedSignatureUI...")
        from advanced_gradio_ui import AdvancedSignatureUI

        print("🎨 Creating UI instance...")
        ui = AdvancedSignatureUI()
        print(f"✅ UI instance created: {type(ui)}")

        print("🎨 Creating interface...")
        interface = ui.create_interface()
        print(f"✅ Interface created: {type(interface)}")

        print("🔍 Checking interface components...")
        # Try to access some attributes to see if interface is properly created
        if hasattr(interface, 'blocks'):
            print(f"✅ Interface has blocks: {len(interface.blocks)} blocks")
        else:
            print("⚠️  Interface doesn't have blocks attribute")

        if hasattr(interface, 'fns'):
            print(f"✅ Interface has functions: {len(interface.fns)} functions")
        else:
            print("⚠️  Interface doesn't have fns attribute")

        print("✅ UI creation test passed!")
        return True

    except Exception as e:
        print(f"❌ UI creation failed: {e}")
        print("📄 Full traceback:")
        traceback.print_exc()
        return False

def test_simple_gradio():
    """Test if basic Gradio functionality works"""
    print("\n🔍 Testing basic Gradio functionality...")
    print("=" * 50)

    try:
        import gradio as gr
        print("✅ Gradio imported successfully")

        # Create a very simple interface
        def greet(name):
            return f"Hello {name}!"

        with gr.Blocks() as demo:
            name = gr.Textbox(label="Name")
            output = gr.Textbox(label="Output")
            greet_btn = gr.Button("Greet")
            greet_btn.click(fn=greet, inputs=name, outputs=output)

        print("✅ Simple Gradio interface created")
        return True

    except Exception as e:
        print(f"❌ Basic Gradio test failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test if model loading works without UI"""
    print("\n🔍 Testing model loading...")
    print("=" * 50)

    try:
        from advanced_signature_detector import AdvancedSignatureDetector

        print("📦 Creating detector...")
        detector = AdvancedSignatureDetector()

        print("🔧 Loading model (this will take time)...")
        detector.load_model()

        print("✅ Model loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("🐛 UI Debug Test Suite")
    print("Testing why the interface appears blank")
    print("=" * 60)

    # Test 1: Basic Gradio
    gradio_ok = test_simple_gradio()

    # Test 2: UI Creation
    ui_ok = test_ui_creation()

    # Test 3: Model Loading
    model_ok = test_model_loading()

    # Summary
    print("\n📊 DEBUG SUMMARY")
    print("=" * 60)
    print(f"Basic Gradio: {'✅ OK' if gradio_ok else '❌ FAILED'}")
    print(f"UI Creation: {'✅ OK' if ui_ok else '❌ FAILED'}")
    print(f"Model Loading: {'✅ OK' if model_ok else '❌ FAILED'}")

    if gradio_ok and ui_ok and model_ok:
        print("\n🎯 All tests passed! The UI should work.")
        print("💡 Try running: python stable_ui_launcher.py")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        print("🔧 Possible solutions:")
        if not gradio_ok:
            print("• Install/reinstall gradio: pip install gradio")
        if not ui_ok:
            print("• Check advanced_gradio_ui.py for errors")
        if not model_ok:
            print("• Model loading issue - try CPU mode: python cpu_ui_launcher.py")

    print("=" * 60)

if __name__ == "__main__":
    main()
