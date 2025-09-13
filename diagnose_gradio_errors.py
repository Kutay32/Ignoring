#!/usr/bin/env python3
"""
Diagnostic script to identify the source of 404 errors in Gradio
"""

import gradio as gr
import sys
import os

def diagnose_gradio():
    """Diagnose Gradio configuration and potential 404 error sources"""
    print("🔍 Gradio Error Diagnosis")
    print("=" * 50)
    
    # Check Gradio version and configuration
    print(f"Gradio version: {gr.__version__}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not in virtual environment")
    
    # Test basic Gradio functionality
    print("\n🧪 Testing basic Gradio functionality...")
    
    def simple_test(text):
        return f"Test successful! Input: {text}"
    
    # Create minimal interface
    with gr.Blocks(
        title="Gradio Error Test",
        theme=gr.themes.Soft(),
        # Try to minimize external dependencies
        css="""
        .gradio-container {
            font-family: Arial, sans-serif;
        }
        """
    ) as demo:
        gr.Markdown("# Gradio Error Diagnosis Test")
        gr.Markdown("This interface tests for 404 errors and external resource loading issues.")
        
        with gr.Row():
            input_text = gr.Textbox(label="Test Input", placeholder="Type something here...")
            output_text = gr.Textbox(label="Test Output", interactive=False)
        
        test_btn = gr.Button("Test Function", variant="primary")
        test_btn.click(simple_test, inputs=input_text, outputs=output_text)
        
        gr.Markdown("""
        ### What to check in browser:
        1. Open browser developer tools (F12)
        2. Go to Console tab
        3. Look for 404 errors or resource loading failures
        4. Check Network tab for failed requests
        """)
    
    print("🚀 Launching diagnostic interface...")
    print("🌐 Open your browser to: http://localhost:7862")
    print("📋 Check browser console for 404 errors")
    print("=" * 50)
    
    try:
        # Launch with minimal configuration to avoid external resources
        demo.launch(
            server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
            server_port=7862,
            share=False,
            show_error=True,
            debug=False,
            quiet=False,
            # Try to minimize external resource loading
            inbrowser=True  # This should open browser automatically
        )
    except Exception as e:
        print(f"❌ Error launching diagnostic interface: {e}")
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Check if port 7862 is available")
        print("2. Try a different port: change server_port=7863")
        print("3. Check firewall settings")
        print("4. Try running as administrator")
        return False
    
    return True

if __name__ == "__main__":
    diagnose_gradio()
