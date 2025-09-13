#!/usr/bin/env python3
"""
Simple UI test with basic functionality
"""

import gradio as gr
import time

def load_model_simple():
    """Simple model loading function"""
    try:
        time.sleep(1)  # Simulate loading
        return "✅ Model loaded successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def process_image_simple(image):
    """Simple image processing"""
    if image is None:
        return "❌ No image uploaded", "Please upload an image first"

    return "✅ Image processed successfully!", f"Image size: {image.size if hasattr(image, 'size') else 'Unknown'}"

def create_simple_interface():
    """Create a very simple test interface"""
    with gr.Blocks(title="Simple Signature Test UI", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🧪 Simple Signature Detection Test UI")

        gr.Markdown("This is a simplified test interface to check if the UI loads properly.")

        with gr.Row():
            with gr.Column():
                model_btn = gr.Button("Load Test Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False, value="Model not loaded")

            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Test Image")
                process_btn = gr.Button("Process Test Image", variant="primary")

        with gr.Row():
            result1 = gr.Textbox(label="Result 1", interactive=False)
            result2 = gr.Textbox(label="Result 2", interactive=False)

        # Event handlers
        model_btn.click(
            load_model_simple,
            outputs=[model_status]
        )

        process_btn.click(
            process_image_simple,
            inputs=[image_input],
            outputs=[result1, result2]
        )

    return interface

def main():
    """Launch the simple test UI"""
    print("🚀 Starting Simple Test UI...")
    print("📋 This UI will test if Gradio works properly")
    print("🌐 UI will be available at: http://localhost:7862")
    print("💡 This uses port 7862 to avoid conflicts")
    print("=" * 50)

    interface = create_simple_interface()

    # Launch with debugging options
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()
