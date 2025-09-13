import gradio as gr
import json
import os
from advanced_signature_detector import AdvancedSignatureDetector
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

class AdvancedSignatureUI:
    def __init__(self):
        self.detector = None
        self.current_model = None
        
    def load_model(self, model_name):
        """Load the specified Qwen model and YOLOv8"""
        try:
            self.detector = AdvancedSignatureDetector(
                model_name=model_name,
                yolo_weights_path="weights/yolov8s.pt",
                use_quantization=True  # Enable quantization for memory efficiency
            )
            self.detector.load_model()
            self.current_model = model_name
            device_info = f"on {self.detector.device}"
            return f"✅ Models loaded successfully!\n  • Qwen: {model_name} {device_info}\n  • YOLOv8: yolov8s.pt"
        except Exception as e:
            return f"❌ Error loading models: {str(e)}"
    
    def process_image(self, image, user_id, similarity_threshold):
        """Process uploaded image for signature detection and analysis"""
        if self.detector is None:
            return "❌ Please load a model first!", None, None, None
        
        if image is None:
            return "❌ Please upload an image!", None, None, None
        
        # Save uploaded image temporarily
        temp_path = f"/tmp/temp_signature_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image.save(temp_path)
        
        try:
            # Process the image
            result = self.detector.process_document_advanced(temp_path, user_id)
            
            if not result["success"]:
                return f"❌ Processing failed: {result['error']}", None, None, None
            
            # Create summary
            summary = f"""
            📊 **Processing Results:**
            
            🔍 **Detection Results:**
            - Total regions detected: {result['total_signatures_detected']}
            - New signatures: {result['new_signatures_count']}
            - Matched signatures: {result['matched_signatures_count']}
            
            📋 **Detected Regions:**
            """
            
            for i, region in enumerate(result['detected_regions']):
                method_display = "YOLOv8 Detection" if region['method'] == 'yolo_detection' else region['method'].replace('_', ' ').title()
                summary += f"""
            Region {i+1}:
            - Method: {method_display}
            - Confidence: {region['confidence']:.2f}
            - Area: {region['area']:.0f} pixels
            - Aspect Ratio: {region['aspect_ratio']:.2f}
            """
            
            # Create detailed analysis
            analysis = "📝 **Detailed Signature Analysis:**\n\n"
            
            for i, sig_result in enumerate(result['signature_results']):
                analysis += f"**Signature {i+1}:**\n"
                analysis += f"- Type: {sig_result['features'].get('signature_type', 'Unknown')}\n"
                analysis += f"- Complexity: {sig_result['features'].get('complexity_level', 'Unknown')}\n"
                analysis += f"- Size: {sig_result['features'].get('size_characteristics', 'Unknown')}\n"
                analysis += f"- Legibility: {sig_result['features'].get('legibility_level', 'Unknown')}\n"
                analysis += f"- Confidence: {sig_result.get('confidence_score', 0):.2f}\n"
                analysis += f"- Detection Method: {sig_result.get('detection_method', 'Unknown')}\n"
                analysis += f"- Hash: {sig_result.get('signature_hash', 'N/A')[:16]}...\n\n"
                
                # Add VLM analysis
                if 'analysis' in sig_result:
                    analysis += f"**VLM Analysis:**\n{sig_result['analysis'][:500]}...\n\n"
            
            # Create similarity report
            similarity_report = "🔍 **Similarity Analysis:**\n\n"
            
            if result['similar_signatures']:
                similarity_report += f"Found {len(result['similar_signatures'])} similar signatures:\n\n"
                
                for i, similar in enumerate(result['similar_signatures'][:5]):  # Show top 5
                    similarity_report += f"**Match {i+1}:**\n"
                    similarity_report += f"- User ID: {similar['user_id']}\n"
                    similarity_report += f"- Overall Score: {similar['overall_score']:.3f}\n"
                    similarity_report += f"- Text Similarity: {similar['similarities']['text_similarity']:.3f}\n"
                    similarity_report += f"- Feature Similarity: {similar['similarities']['feature_similarity']:.3f}\n"
                    similarity_report += f"- Confidence: {similar['confidence_score']:.2f}\n"
                    similarity_report += f"- Date: {similar['timestamp']}\n\n"
            else:
                similarity_report += "No similar signatures found in database.\n"
            
            # Create visualization data
            viz_data = self._create_visualization_data(result)
            
            return summary, analysis, similarity_report, viz_data
            
        except Exception as e:
            return f"❌ Error processing image: {str(e)}", None, None, None
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _create_visualization_data(self, result):
        """Create data for visualization"""
        if not result['signature_results']:
            return None
        
        # Create DataFrame for visualization
        data = []
        for i, sig in enumerate(result['signature_results']):
            data.append({
                'Signature': f"Signature {i+1}",
                'Type': sig['features'].get('signature_type', 'Unknown'),
                'Complexity': sig['features'].get('complexity_level', 'Unknown'),
                'Confidence': sig.get('confidence_score', 0),
                'Detection_Confidence': sig.get('detection_confidence', 0)
            })
        
        return pd.DataFrame(data)
    
    def compare_signatures(self, signature1_id, signature2_id):
        """Compare two signatures by ID"""
        if self.detector is None:
            return "❌ Please load a model first!"
        
        try:
            report = self.detector.generate_comparison_report(int(signature1_id), int(signature2_id))
            
            if "error" in report:
                return f"❌ {report['error']}"
            
            comparison_text = f"""
            🔍 **Signature Comparison Report**
            
            **Signature 1:**
            - ID: {report['signature1']['id']}
            - User: {report['signature1']['user_id']}
            - Type: {report['signature1']['signature_type']}
            - Confidence: {report['signature1']['confidence_score']:.2f}
            
            **Signature 2:**
            - ID: {report['signature2']['id']}
            - User: {report['signature2']['user_id']}
            - Type: {report['signature2']['signature_type']}
            - Confidence: {report['signature2']['confidence_score']:.2f}
            
            **Similarity Analysis:**
            - Overall Similarity: {report['similarities']['overall_similarity']:.3f}
            - Text Similarity: {report['similarities']['text_similarity']:.3f}
            - Feature Similarity: {report['similarities']['feature_similarity']:.3f}
            
            **Verdict: {report['verdict']}**
            - Confidence: {report['confidence']:.2f}
            """
            
            return comparison_text
            
        except Exception as e:
            return f"❌ Error comparing signatures: {str(e)}"
    
    def get_database_stats(self):
        """Get database statistics"""
        if self.detector is None:
            return "❌ Please load a model first!"
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.detector.db_path)
            cursor = conn.cursor()
            
            # Get signature count
            cursor.execute('SELECT COUNT(*) FROM signatures')
            total_signatures = cursor.fetchone()[0]
            
            # Get signature types
            cursor.execute('SELECT signature_type, COUNT(*) FROM signatures GROUP BY signature_type')
            type_counts = cursor.fetchall()
            
            # Get recent signatures
            cursor.execute('SELECT user_id, signature_type, timestamp FROM signatures ORDER BY timestamp DESC LIMIT 10')
            recent_signatures = cursor.fetchall()
            
            conn.close()
            
            stats = f"""
            📊 **Database Statistics**
            
            **Total Signatures:** {total_signatures}
            
            **Signature Types:**
            """
            
            for sig_type, count in type_counts:
                stats += f"- {sig_type}: {count}\n"
            
            stats += "\n**Recent Signatures:**\n"
            for user_id, sig_type, timestamp in recent_signatures:
                stats += f"- {user_id} ({sig_type}) - {timestamp}\n"
            
            return stats
            
        except Exception as e:
            return f"❌ Error getting database stats: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="Advanced Signature Detection & Comparison System", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🔍 Advanced Signature Detection & Comparison System

            This system uses **YOLOv8 object detection** and **Qwen Vision Language Models** for:
            - **YOLOv8-powered signature detection** with OpenCV fallback
            - **Advanced feature extraction** using Qwen VLM analysis
            - **Comprehensive similarity comparison** with multiple metrics
            - **Signature verification and matching**
            """)
            
            with gr.Tabs():
                # Main Processing Tab
                with gr.Tab("🔍 Signature Detection"):
                    with gr.Row():
                        with gr.Column():
                            model_dropdown = gr.Dropdown(
                                choices=[
                                    "Qwen/Qwen2.5-VL-7B-Instruct"
                                ],
                                value="Qwen/Qwen2.5-VL-7B-Instruct",
                                label="Select Model"
                            )
                            load_model_btn = gr.Button("Load Model", variant="primary")
                            model_status = gr.Textbox(label="Model Status", interactive=False)
                        
                        with gr.Column():
                            image_input = gr.Image(type="pil", label="Upload Document Image")
                            user_id_input = gr.Textbox(label="User ID (Optional)", placeholder="Enter user ID for storage")
                            threshold_input = gr.Slider(0.1, 1.0, 0.7, label="Similarity Threshold")
                            process_btn = gr.Button("Process Image", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            summary_output = gr.Textbox(label="Processing Summary", lines=10, interactive=False)
                        with gr.Column():
                            analysis_output = gr.Textbox(label="Detailed Analysis", lines=10, interactive=False)
                    
                    with gr.Row():
                        similarity_output = gr.Textbox(label="Similarity Report", lines=10, interactive=False)
                
                # Comparison Tab
                with gr.Tab("🔄 Signature Comparison"):
                    with gr.Row():
                        with gr.Column():
                            sig1_id = gr.Number(label="Signature 1 ID", value=1)
                            sig2_id = gr.Number(label="Signature 2 ID", value=2)
                            compare_btn = gr.Button("Compare Signatures", variant="primary")
                        with gr.Column():
                            comparison_output = gr.Textbox(label="Comparison Report", lines=15, interactive=False)
                
                # Database Tab
                with gr.Tab("📊 Database Management"):
                    with gr.Row():
                        with gr.Column():
                            db_stats_btn = gr.Button("Get Database Statistics", variant="primary")
                            db_stats_output = gr.Textbox(label="Database Statistics", lines=15, interactive=False)
                
                # Visualization Tab
                with gr.Tab("📈 Visualizations"):
                    gr.Markdown("### Signature Analysis Visualizations")
                    viz_output = gr.Plot(label="Signature Analysis Chart")
            
            # Event handlers
            load_model_btn.click(
                self.load_model,
                inputs=[model_dropdown],
                outputs=[model_status]
            )
            
            process_btn.click(
                self.process_image,
                inputs=[image_input, user_id_input, threshold_input],
                outputs=[summary_output, analysis_output, similarity_output, viz_output]
            )
            
            compare_btn.click(
                self.compare_signatures,
                inputs=[sig1_id, sig2_id],
                outputs=[comparison_output]
            )
            
            db_stats_btn.click(
                self.get_database_stats,
                outputs=[db_stats_output]
            )
        
        return interface

def main():
    """Main function to launch the interface"""
    ui = AdvancedSignatureUI()
    interface = ui.create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()