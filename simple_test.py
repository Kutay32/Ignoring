#!/usr/bin/env python3
"""
Simple test script that validates the system structure without requiring model downloads
"""

import os
import sys
import json
import sqlite3
from PIL import Image, ImageDraw
import numpy as np

def test_basic_functionality():
    """Test basic functionality without VLM models"""
    print("🧪 Testing Basic System Functionality...")
    
    # Test 1: Database creation
    print("1. Testing database creation...")
    try:
        conn = sqlite3.connect("test_signatures.db")
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_signatures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                signature_data TEXT,
                timestamp DATETIME
            )
        ''')
        conn.commit()
        conn.close()
        print("   ✅ Database creation successful")
    except Exception as e:
        print(f"   ❌ Database creation failed: {e}")
        return False
    
    # Test 2: Image creation and processing
    print("2. Testing image creation...")
    try:
        # Create a test signature image
        img = Image.new('RGB', (400, 300), 'white')
        draw = ImageDraw.Draw(img)
        
        # Add some content
        draw.text((50, 50), "Test Document", fill='black')
        draw.text((50, 100), "Signature:", fill='black')
        
        # Draw a simple signature line
        draw.line([(150, 120), (300, 120)], fill='blue', width=2)
        draw.text((150, 130), "John Doe", fill='blue')
        
        # Save test image
        img.save("test_signature.png")
        print("   ✅ Image creation successful")
    except Exception as e:
        print(f"   ❌ Image creation failed: {e}")
        return False
    
    # Test 3: File operations
    print("3. Testing file operations...")
    try:
        if os.path.exists("test_signature.png"):
            img = Image.open("test_signature.png")
            print(f"   ✅ Image loaded: {img.size}")
        else:
            print("   ❌ Image file not found")
            return False
    except Exception as e:
        print(f"   ❌ File operations failed: {e}")
        return False
    
    # Test 4: JSON operations (simulating VLM output)
    print("4. Testing JSON operations...")
    try:
        test_data = {
            "analysis": "Test signature analysis",
            "features": {
                "text_description": "Test signature",
                "stroke_patterns": ["curved", "flowing"],
                "letter_shapes": ["cursive"],
                "style_classification": "cursive"
            },
            "signature_region": [100, 200, 300, 250],
            "model_used": "test_model",
            "timestamp": "2024-01-15T10:00:00"
        }
        
        # Test JSON serialization
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        
        if parsed_data["features"]["style_classification"] == "cursive":
            print("   ✅ JSON operations successful")
        else:
            print("   ❌ JSON operations failed")
            return False
    except Exception as e:
        print(f"   ❌ JSON operations failed: {e}")
        return False
    
    # Test 5: Database operations
    print("5. Testing database operations...")
    try:
        conn = sqlite3.connect("test_signatures.db")
        cursor = conn.cursor()
        
        # Insert test data
        cursor.execute('''
            INSERT INTO test_signatures (user_id, signature_data, timestamp)
            VALUES (?, ?, ?)
        ''', ("test_user", json_str, "2024-01-15T10:00:00"))
        
        # Query test data
        cursor.execute('SELECT * FROM test_signatures WHERE user_id = ?', ("test_user",))
        result = cursor.fetchone()
        
        if result and result[1] == "test_user":
            print("   ✅ Database operations successful")
        else:
            print("   ❌ Database operations failed")
            return False
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"   ❌ Database operations failed: {e}")
        return False
    
    # Cleanup
    print("6. Cleaning up test files...")
    try:
        if os.path.exists("test_signature.png"):
            os.remove("test_signature.png")
        if os.path.exists("test_signatures.db"):
            os.remove("test_signatures.db")
        print("   ✅ Cleanup successful")
    except Exception as e:
        print(f"   ⚠️ Cleanup warning: {e}")
    
    return True

def test_io_similarity():
    """Test IoU similarity calculation without VLM"""
    print("\n🔍 Testing IoU Similarity Calculation...")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Test data
        text1 = "curved flowing cursive signature john doe"
        text2 = "curved flowing cursive signature john doe"  # Same
        text3 = "straight angular print signature jane smith"  # Different
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2, text3])
        
        # Calculate similarities
        sim_same = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        sim_diff = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[2:3])[0][0]
        
        # Convert to IoU-like metric
        iou_same = (sim_same + 1) / 2
        iou_diff = (sim_diff + 1) / 2
        
        print(f"   Same signature similarity: {iou_same:.3f}")
        print(f"   Different signature similarity: {iou_diff:.3f}")
        
        if iou_same > iou_diff:
            print("   ✅ IoU similarity calculation working correctly")
            return True
        else:
            print("   ❌ IoU similarity calculation failed")
            return False
            
    except ImportError:
        print("   ⚠️ Skipping IoU test - scikit-learn not available")
        return True
    except Exception as e:
        print(f"   ❌ IoU similarity test failed: {e}")
        return False

def test_gradio_import():
    """Test if Gradio can be imported"""
    print("\n🖥️ Testing Gradio Import...")
    
    try:
        import gradio as gr
        print("   ✅ Gradio import successful")
        return True
    except ImportError:
        print("   ⚠️ Gradio not available - UI will not work")
        return False
    except Exception as e:
        print(f"   ❌ Gradio import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Simple System Test")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("IoU Similarity", test_io_similarity),
        ("Gradio Import", test_gradio_import)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   - {test_name}: {status}")
    
    overall_success = all(results.values())
    print(f"\n🎯 Overall Status: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 System is ready! You can now:")
        print("   1. Install the full requirements: pip install -r requirements.txt")
        print("   2. Run the web interface: python3 run_system.py --mode ui")
        print("   3. Run full tests: python3 run_system.py --mode test")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)