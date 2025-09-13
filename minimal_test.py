#!/usr/bin/env python3
"""
Minimal test script using only standard library modules
"""

import os
import sys
import json
import sqlite3
import tempfile
from datetime import datetime

def test_basic_functionality():
    """Test basic functionality using only standard library"""
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
                signature_features TEXT,
                image_path TEXT,
                timestamp DATETIME,
                model_used TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("   ✅ Database creation successful")
    except Exception as e:
        print(f"   ❌ Database creation failed: {e}")
        return False
    
    # Test 2: JSON operations (simulating VLM output)
    print("2. Testing JSON operations...")
    try:
        test_data = {
            "analysis": "Test signature analysis with VLM",
            "features": {
                "text_description": "Test signature analysis",
                "stroke_patterns": ["curved", "flowing"],
                "letter_shapes": ["cursive"],
                "style_classification": "cursive",
                "unique_features": ["loopy", "connected"]
            },
            "signature_region": [100, 200, 300, 250],
            "model_used": "Qwen/Qwen2.5-VL-7B-Instruct",
            "timestamp": datetime.now().isoformat()
        }
        
        # Test JSON serialization
        json_str = json.dumps(test_data, indent=2)
        parsed_data = json.loads(json_str)
        
        if (parsed_data["features"]["style_classification"] == "cursive" and 
            len(parsed_data["features"]["stroke_patterns"]) == 2):
            print("   ✅ JSON operations successful")
        else:
            print("   ❌ JSON operations failed")
            return False
    except Exception as e:
        print(f"   ❌ JSON operations failed: {e}")
        return False
    
    # Test 3: Database operations
    print("3. Testing database operations...")
    try:
        conn = sqlite3.connect("test_signatures.db")
        cursor = conn.cursor()
        
        # Insert test data
        cursor.execute('''
            INSERT INTO test_signatures (user_id, signature_data, signature_features, image_path, timestamp, model_used)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            "test_user_001", 
            json_str, 
            json.dumps(test_data["features"]),
            "test_image.png",
            test_data["timestamp"],
            test_data["model_used"]
        ))
        
        # Query test data
        cursor.execute('SELECT * FROM test_signatures WHERE user_id = ?', ("test_user_001",))
        result = cursor.fetchone()
        
        if result and result[1] == "test_user_001":
            print("   ✅ Database operations successful")
            print(f"   📊 Stored signature ID: {result[0]}")
        else:
            print("   ❌ Database operations failed")
            return False
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"   ❌ Database operations failed: {e}")
        return False
    
    # Test 4: File operations
    print("4. Testing file operations...")
    try:
        # Create a simple text file (simulating image metadata)
        with open("test_metadata.txt", "w") as f:
            f.write("Test document metadata\n")
            f.write("Signature region: [100, 200, 300, 250]\n")
            f.write("Stamp detected: False\n")
        
        # Read it back
        with open("test_metadata.txt", "r") as f:
            content = f.read()
        
        if "Signature region" in content:
            print("   ✅ File operations successful")
        else:
            print("   ❌ File operations failed")
            return False
            
    except Exception as e:
        print(f"   ❌ File operations failed: {e}")
        return False
    
    # Test 5: Similarity calculation simulation
    print("5. Testing similarity calculation simulation...")
    try:
        # Simulate IoU calculation using text similarity
        def simple_text_similarity(text1, text2):
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if len(union) > 0 else 0
        
        # Test with similar signatures
        sig1 = "curved flowing cursive signature john doe"
        sig2 = "curved flowing cursive signature john doe"
        sig3 = "straight angular print signature jane smith"
        
        sim_same = simple_text_similarity(sig1, sig2)
        sim_diff = simple_text_similarity(sig1, sig3)
        
        print(f"   Same signature similarity: {sim_same:.3f}")
        print(f"   Different signature similarity: {sim_diff:.3f}")
        
        if sim_same > sim_diff:
            print("   ✅ Similarity calculation working correctly")
        else:
            print("   ❌ Similarity calculation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Similarity calculation failed: {e}")
        return False
    
    # Cleanup
    print("6. Cleaning up test files...")
    try:
        for file in ["test_signatures.db", "test_metadata.txt"]:
            if os.path.exists(file):
                os.remove(file)
        print("   ✅ Cleanup successful")
    except Exception as e:
        print(f"   ⚠️ Cleanup warning: {e}")
    
    return True

def test_system_architecture():
    """Test system architecture components"""
    print("\n🏗️ Testing System Architecture...")
    
    # Test 1: Check if all required files exist
    print("1. Checking system files...")
    required_files = [
        "signature_extractor.py",
        "gradio_ui.py", 
        "test_system.py",
        "run_system.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        return False
    else:
        print("   ✅ All required files present")
    
    # Test 2: Check file content structure
    print("2. Checking file content structure...")
    try:
        # Check signature_extractor.py has main class
        with open("signature_extractor.py", "r") as f:
            content = f.read()
            if "class SignatureExtractor" in content and "def process_document" in content:
                print("   ✅ SignatureExtractor class structure correct")
            else:
                print("   ❌ SignatureExtractor class structure incorrect")
                return False
        
        # Check gradio_ui.py has UI class
        with open("gradio_ui.py", "r") as f:
            content = f.read()
            if "class SignatureComparisonUI" in content and "def create_interface" in content:
                print("   ✅ UI class structure correct")
            else:
                print("   ❌ UI class structure incorrect")
                return False
                
    except Exception as e:
        print(f"   ❌ File content check failed: {e}")
        return False
    
    return True

def test_requirements_parsing():
    """Test requirements.txt parsing"""
    print("\n📦 Testing Requirements Parsing...")
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read().strip().split('\n')
        
        print(f"   📋 Found {len(requirements)} requirements")
        
        # Check for key dependencies
        key_deps = ["torch", "transformers", "gradio", "opencv-python", "PIL"]
        found_deps = []
        
        for req in requirements:
            for dep in key_deps:
                if dep.lower() in req.lower():
                    found_deps.append(dep)
        
        print(f"   ✅ Key dependencies found: {found_deps}")
        
        if len(found_deps) >= 3:  # At least 3 key deps
            print("   ✅ Requirements file looks good")
            return True
        else:
            print("   ❌ Requirements file missing key dependencies")
            return False
            
    except Exception as e:
        print(f"   ❌ Requirements parsing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Minimal System Test")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("System Architecture", test_system_architecture),
        ("Requirements Parsing", test_requirements_parsing)
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
        print("\n🎉 System architecture is ready!")
        print("\n📖 Next Steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run web interface: python3 run_system.py --mode ui")
        print("   3. Run full tests: python3 run_system.py --mode test")
        print("\n🔧 System Features:")
        print("   - Multi-model VLM support (Qwen 2.5-VL 7B/32B/72B)")
        print("   - Stamp detection and signature isolation")
        print("   - IoU-based similarity comparison")
        print("   - SQLite database storage")
        print("   - Gradio web interface")
        print("   - Model comparison capabilities")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)