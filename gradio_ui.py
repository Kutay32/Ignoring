import gradio as gr
import os
import json
from signature_extractor import SignatureExtractor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tempfile

class SignatureComparisonUI:
    def __init__(self):
        self.models = {
            "Qwen2.5-VL-7B": "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen2.5-VL-32B": "Qwen/Qwen2.5-VL-32B-Instruct", 
            "Qwen2.5-VL-72B": "Qwen/Qwen2.5-VL-72B-Instruct"
        }
        self.extractors = {}
        self.results_history = []
        
    def load_model(self, model_name):
        """Load the selected model"""
        if model_name not in self.extractors:
            try:
                extractor = SignatureExtractor(self.models[model_name])
                extractor.load_model()
                self.extractors[model_name] = extractor
                return f"✅ {model_name} loaded successfully!"
            except Exception as e:
                return f"❌ Error loading {model_name}: {str(e)}"
        else:
            return f"✅ {model_name} already loaded!"
    
    def process_image(self, image, model_name, user_id, threshold):
        """Process uploaded image with selected model"""
        if image is None:
            return "Please upload an image first.", None, None, None
        
        if model_name not in self.extractors:
            return "Please load the model first.", None, None, None
        
        try:
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image.save(tmp_file.name)
                temp_path = tmp_file.name
            
            # Process with selected model
            extractor = self.extractors[model_name]
            result = extractor.process_document(temp_path, user_id if user_id else None)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            if not result["success"]:
                return f"Error processing image: {result['error']}", None, None, None
            
            # Create visualization
            vis_image = self.create_visualization(result, image)
            
            # Format results
            summary = self.format_results(result, model_name)
            
            # Store in history
            self.results_history.append({
                "model": model_name,
                "result": result,
                "timestamp": result["signature_data"]["timestamp"] if result["signature_data"] else None
            })
            
            return summary, vis_image, result, None
            
        except Exception as e:
            return f"Error processing image: {str(e)}", None, None, None
    
    def create_visualization(self, result, original_image):
        """Create visualization of detected regions and results"""
        if not result["regions"] or not result["regions"]["full_image"]:
            return original_image
        
        # Convert PIL to numpy array if needed
        if hasattr(original_image, 'convert'):
            image = np.array(original_image.convert('RGB'))
        else:
            image = np.array(original_image)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with regions
        axes[0].imshow(image)
        axes[0].set_title("Detected Regions")
        axes[0].axis('off')
        
        # Draw regions if detected
        if result["regions"]["stamp_region"]:
            x1, y1, x2, y2 = result["regions"]["stamp_region"]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].text(x1, y1-10, "STAMP", color='red', fontsize=12, fontweight='bold')
        
        if result["regions"]["signature_region"]:
            x1, y1, x2, y2 = result["regions"]["signature_region"]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='green', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].text(x1, y1-10, "SIGNATURE", color='green', fontsize=12, fontweight='bold')
        
        # Results summary
        axes[1].text(0.1, 0.9, f"Model: {result.get('model_used', 'Unknown')}", fontsize=14, fontweight='bold')
        axes[1].text(0.1, 0.8, f"Status: {'NEW SIGNATURE' if result['is_new'] else 'MATCH FOUND'}", 
                    fontsize=12, color='green' if result['is_new'] else 'orange')
        axes[1].text(0.1, 0.7, f"IoU Score: {result['iou_score']:.3f}", fontsize=12)
        
        if result["best_match"]:
            axes[1].text(0.1, 0.6, f"Best Match User: {result['best_match']['user_id']}", fontsize=12)
            axes[1].text(0.1, 0.5, f"Similarity: {result['best_match']['similarity']:.3f}", fontsize=12)
        
        axes[1].text(0.1, 0.4, f"Similar Signatures: {len(result['similar_signatures'])}", fontsize=12)
        axes[1].text(0.1, 0.3, f"Threshold: {result.get('threshold', 0.7)}", fontsize=12)
        
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        fig.canvas.draw()
        vis_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)
        
        return vis_image
    
    def format_results(self, result, model_name):
        """Format results for display"""
        summary = f"""
## Signature Analysis Results

**Model Used:** {model_name}
**Processing Status:** {'✅ Success' if result['success'] else '❌ Failed'}

### Detection Results:
- **Stamp Detected:** {'Yes' if result['regions'] and result['regions']['stamp_region'] else 'No'}
- **Signature Detected:** {'Yes' if result['regions'] and result['regions']['signature_region'] else 'No'}

### Signature Analysis:
- **Status:** {'🆕 NEW SIGNATURE' if result['is_new'] else '🔄 MATCH FOUND'}
- **IoU Similarity Score:** {result['iou_score']:.3f}
- **Similar Signatures Found:** {len(result['similar_signatures'])}

"""
        
        if result['best_match']:
            summary += f"""
### Best Match:
- **User ID:** {result['best_match']['user_id']}
- **Similarity Score:** {result['best_match']['similarity']:.3f}
- **Timestamp:** {result['best_match']['timestamp']}
"""
        
        if result['signature_data'] and result['signature_data']['features']:
            features = result['signature_data']['features']
            summary += f"""
### Signature Features:
- **Style:** {features.get('style_classification', 'Unknown')}
- **Stroke Patterns:** {', '.join(features.get('stroke_patterns', []))}
- **Letter Shapes:** {', '.join(features.get('letter_shapes', []))}
"""
        
        return summary
    
    def compare_models(self, image, user_id, threshold):
        """Compare results across all loaded models"""
        if image is None:
            return "Please upload an image first.", None
        
        if not self.extractors:
            return "Please load at least one model first.", None
        
        comparison_results = []
        
        for model_name, extractor in self.extractors.items():
            try:
                # Save uploaded image temporarily
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    image.save(tmp_file.name)
                    temp_path = tmp_file.name
                
                result = extractor.process_document(temp_path, user_id if user_id else None)
                os.unlink(temp_path)
                
                comparison_results.append({
                    "model": model_name,
                    "iou_score": result['iou_score'],
                    "is_new": result['is_new'],
                    "similar_count": len(result['similar_signatures']),
                    "success": result['success']
                })
                
            except Exception as e:
                comparison_results.append({
                    "model": model_name,
                    "iou_score": 0.0,
                    "is_new": True,
                    "similar_count": 0,
                    "success": False,
                    "error": str(e)
                })
        
        # Create comparison visualization
        vis = self.create_comparison_visualization(comparison_results)
        
        # Format comparison results
        summary = self.format_comparison_results(comparison_results)
        
        return summary, vis
    
    def create_comparison_visualization(self, results):
        """Create comparison visualization across models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = [r["model"] for r in results]
        iou_scores = [r["iou_score"] for r in results]
        similar_counts = [r["similar_count"] for r in results]
        
        # IoU Scores comparison
        bars1 = ax1.bar(models, iou_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('IoU Similarity Scores by Model')
        ax1.set_ylabel('IoU Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars1, iou_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Similar signatures count
        bars2 = ax2.bar(models, similar_counts, color=['orange', 'purple', 'brown'])
        ax2.set_title('Similar Signatures Found by Model')
        ax2.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, count in zip(bars2, similar_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        fig.canvas.draw()
        vis_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)
        
        return vis_image
    
    def format_comparison_results(self, results):
        """Format comparison results for display"""
        summary = "## Model Comparison Results\n\n"
        
        for result in results:
            status = "✅ Success" if result["success"] else "❌ Failed"
            summary += f"### {result['model']}\n"
            summary += f"- **Status:** {status}\n"
            summary += f"- **IoU Score:** {result['iou_score']:.3f}\n"
            summary += f"- **New Signature:** {'Yes' if result['is_new'] else 'No'}\n"
            summary += f"- **Similar Signatures:** {result['similar_count']}\n"
            if not result["success"] and "error" in result:
                summary += f"- **Error:** {result['error']}\n"
            summary += "\n"
        
        return summary
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Signature Extraction & Comparison System", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🔍 Document Signature Extraction & Comparison System")
            gr.Markdown("Extract and compare document signatures using Qwen 2.5-VL models with IoU similarity metrics")
            
            with gr.Tab("Single Model Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            choices=list(self.models.keys()),
                            label="Select Model",
                            value=list(self.models.keys())[0]
                        )
                        load_btn = gr.Button("Load Model", variant="primary")
                        load_status = gr.Textbox(label="Model Status", interactive=False)
                        
                        user_id_input = gr.Textbox(
                            label="User ID (optional)",
                            placeholder="Enter user ID for storage"
                        )
                        threshold_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.05,
                            label="Similarity Threshold"
                        )
                        
                        image_input = gr.Image(
                            label="Upload Document Image",
                            type="pil"
                        )
                        process_btn = gr.Button("Process Image", variant="primary")
                    
                    with gr.Column(scale=2):
                        results_text = gr.Markdown(label="Analysis Results")
                        visualization = gr.Image(label="Visualization")
            
            with gr.Tab("Model Comparison"):
                with gr.Row():
                    with gr.Column(scale=1):
                        comp_user_id = gr.Textbox(
                            label="User ID (optional)",
                            placeholder="Enter user ID for storage"
                        )
                        comp_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.05,
                            label="Similarity Threshold"
                        )
                        comp_image_input = gr.Image(
                            label="Upload Document Image",
                            type="pil"
                        )
                        compare_btn = gr.Button("Compare All Models", variant="primary")
                    
                    with gr.Column(scale=2):
                        comparison_results = gr.Markdown(label="Comparison Results")
                        comparison_viz = gr.Image(label="Comparison Visualization")
            
            with gr.Tab("Model Management"):
                gr.Markdown("### Load Models")
                with gr.Row():
                    for model_name in self.models.keys():
                        with gr.Column():
                            gr.Markdown(f"**{model_name}**")
                            load_btn = gr.Button(f"Load {model_name}", variant="secondary")
                            load_btn.click(
                                lambda m=model_name: self.load_model(m),
                                outputs=gr.Textbox(visible=False)
                            )
            
            # Event handlers
            load_btn.click(
                self.load_model,
                inputs=[model_dropdown],
                outputs=[load_status]
            )
            
            process_btn.click(
                self.process_image,
                inputs=[image_input, model_dropdown, user_id_input, threshold_slider],
                outputs=[results_text, visualization, gr.State(), gr.State()]
            )
            
            compare_btn.click(
                self.compare_models,
                inputs=[comp_image_input, comp_user_id, comp_threshold],
                outputs=[comparison_results, comparison_viz]
            )
        
        return interface

def main():
    """Main function to launch the interface"""
    ui = SignatureComparisonUI()
    interface = ui.create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    main()