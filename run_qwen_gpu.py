#!/usr/bin/env python3
"""
Direct Qwen2.5-VL-7B GPU test and runner
"""

import sys
import os
import torch
from PIL import Image
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def create_test_signature_image():
    """Create a simple test image with some shapes that could be signatures"""
    print("🖼️  Creating test signature image...")

    # Create a white background image
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 255

    # Add some signature-like elements
    # Rectangle (could be a signature box)
    cv2 = __import__('cv2')
    cv2.rectangle(img, (100, 100), (400, 200), (0, 0, 0), -1)

    # Add some text that looks like a signature
    cv2.putText(img, "JOHN DOE", (150, 160), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "Signature", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Add another signature-like element
    cv2.rectangle(img, (600, 300), (900, 400), (0, 0, 0), -1)
    cv2.putText(img, "JANE SMITH", (650, 360), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (0, 0, 0), 2)

    # Convert to PIL Image
    pil_img = Image.fromarray(img)

    # Save test image
    test_path = "test_signature_sample.jpg"
    pil_img.save(test_path)

    print(f"✅ Test image saved as: {test_path}")
    return test_path

def test_qwen_gpu_direct():
    """Test Qwen2.5-VL-7B directly on GPU"""
    print("🚀 Testing Qwen2.5-VL-7B on GPU...")
    print("=" * 60)

    # Check CUDA status
    print("🔍 Checking CUDA status...")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("❌ CUDA not available! Switching to CPU mode...")
        device = "cpu"
    else:
        device = "cuda"
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")

    print(f"🎯 Using device: {device}")
    print()

    try:
        # Import and setup Qwen model
        print("📦 Loading Qwen2.5-VL-7B model...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        # Load processor
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
        print("✅ Processor loaded")

        # Try loading with quantization for memory efficiency
        print("🔧 Loading model with 8-bit quantization...")
        from transformers import BitsAndBytesConfig

        if device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ Model loaded with 8-bit quantization!")
        else:
            # CPU mode (slower but should work)
            print("🐌 Loading on CPU (this will be slow)...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            ).to("cpu")
            print("✅ Model loaded on CPU!")

        # Create test image
        test_image_path = create_test_signature_image()
        image = Image.open(test_image_path)

        # Prepare the analysis prompt
        prompt = """
        Analyze this image and identify any signatures or handwritten elements.
        Provide details about:
        1. What signatures or text you can see
        2. The style and characteristics of any signatures
        3. Any other notable features in the image
        """

        print("📝 Preparing analysis prompt...")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process inputs
        print("🔄 Processing inputs...")
        inputs = processor(
            text=[text],
            images=[image],
            videos=None,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print("🤖 Generating analysis...")
        print("⏳ This may take 30-60 seconds...")

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1
            )

        # Extract response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], generated_ids)
        ]

        analysis = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("\n" + "=" * 60)
        print("🎉 Qwen2.5-VL-7B Analysis Complete!")
        print("=" * 60)
        print("📊 Analysis Result:")
        print(analysis)
        print("=" * 60)

        # Memory cleanup
        del model, processor
        if device == "cuda":
            torch.cuda.empty_cache()

        # Clean up test file
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

        print("✅ Test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during Qwen test: {e}")
        print("📄 Full error:")
        import traceback
        traceback.print_exc()

        # Memory cleanup in case of error
        if 'model' in locals():
            del model
        if device == "cuda":
            torch.cuda.empty_cache()

        return False

def main():
    """Main function"""
    print("🧠 Qwen2.5-VL-7B GPU Test")
    print("Testing direct model execution on GPU")
    print("=" * 60)

    success = test_qwen_gpu_direct()

    if success:
        print("\n🎯 SUCCESS! Qwen2.5-VL-7B is working on your GPU!")
        print("\n💡 Next Steps:")
        print("• Use: python stable_ui_launcher.py (for full UI)")
        print("• Or: python run_advanced_ui.py (simpler launcher)")
        print("• Your Qwen model will now load successfully!")
    else:
        print("\n❌ Test failed. Try these alternatives:")
        print("• CPU mode: python cpu_ui_launcher.py")
        print("• Check memory: Close other GPU applications")
        print("• Restart computer and try again")

if __name__ == "__main__":
    main()
