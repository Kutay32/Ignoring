#!/usr/bin/env python3
"""
Basic Gradio test to diagnose the 404 errors
"""

import gradio as gr
import sys

def test_gradio():
    """Test basic Gradio functionality without external dependencies"""
    print("🔍 Testing Gradio configuration...")
    print(f"Gradio version: {gr.__version__}")
    print(f"Python version: {sys.version}")
    
    def simple_function(text):
        return f"Hello! You said: {text}"
    
    # Create a minimal interface
    with gr.Blocks(title="Gradio Test", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Simple Gradio Test")
        
        with gr.Row():
            text_input = gr.Textbox(label="Enter some text")
            text_output = gr.Textbox(label="Output")
        
        submit_btn = gr.Button("Submit")
        submit_btn.click(simple_function, inputs=text_input, outputs=text_output)
    
    print("🚀 Launching test interface...")
    print("📋 This will help diagnose the 404 errors")
    print("🌐 Interface will be available at: http://localhost:7861")
    print("=" * 50)
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            show_error=True,
            debug=False,
            quiet=False
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_gradio()
