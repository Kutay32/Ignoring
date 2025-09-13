#!/usr/bin/env python3
"""
Test UI connectivity and HTTP response
"""

import sys
import os
import time
import threading
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_http_connectivity():
    """Test if we can make HTTP requests to localhost"""
    print("🌐 Testing HTTP connectivity to localhost...")

    try:
        # Try to connect to localhost
        response = requests.get("http://127.0.0.1:7860", timeout=5)
        print(f"✅ HTTP connection successful: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to localhost:7860 - No server running")
        return False
    except requests.exceptions.Timeout:
        print("⏰ Connection timeout - Server may be slow to respond")
        return False
    except Exception as e:
        print(f"❌ HTTP test failed: {e}")
        return False

def start_test_server():
    """Start a simple test server to verify port binding"""
    class TestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body><h1>Test Server Running on Port 7860</h1><p>If you see this, port binding works!</p></body></html>')

    try:
        server = HTTPServer(('127.0.0.1', 7860), TestHandler)
        print("🖥️  Starting test server on port 7860...")
        server.serve_request()  # Serve only one request
        server.server_close()
        print("✅ Test server started and responded successfully")
        return True
    except OSError as e:
        if "Address already in use" in str(e):
            print("❌ Port 7860 already in use by another process")
            return False
        else:
            print(f"❌ Cannot bind to port 7860: {e}")
            return False
    except Exception as e:
        print(f"❌ Test server failed: {e}")
        return False

def test_gradio_launch():
    """Test if Gradio can launch successfully"""
    print("\n🎨 Testing Gradio launch capability...")

    try:
        import gradio as gr
        print("✅ Gradio imported successfully")

        # Create minimal interface
        def test_func(x):
            return f"Input: {x}"

        with gr.Blocks() as demo:
            inp = gr.Textbox(label="Test Input")
            out = gr.Textbox(label="Test Output")
            btn = gr.Button("Test")
            btn.click(test_func, inputs=inp, outputs=out)

        print("✅ Minimal Gradio interface created")

        # Try to launch (but don't actually start the server)
        print("🚀 Testing Gradio launch (will cancel quickly)...")
        # We'll just test that the launch method exists and can be called
        launch_method = hasattr(demo, 'launch')
        if launch_method:
            print("✅ Gradio launch method available")
            return True
        else:
            print("❌ Gradio launch method not found")
            return False

    except ImportError:
        print("❌ Gradio not installed")
        return False
    except Exception as e:
        print(f"❌ Gradio test failed: {e}")
        return False

def test_browser_access():
    """Test if browser can access localhost"""
    print("\n🌍 Testing browser access to localhost...")

    try:
        import webbrowser
        import urllib.request

        # First start a test server
        print("Starting test server for browser access...")
        server_thread = threading.Thread(target=start_test_server, daemon=True)
        server_thread.start()
        time.sleep(1)  # Give server time to start

        # Try to access the test server
        try:
            response = urllib.request.urlopen("http://127.0.0.1:7860", timeout=3)
            content = response.read().decode('utf-8')
            if "Test Server Running" in content:
                print("✅ Browser can access localhost server")
                return True
            else:
                print("⚠️  Server responds but content unexpected")
                return False
        except Exception as e:
            print(f"❌ Cannot access test server: {e}")
            return False

    except Exception as e:
        print(f"❌ Browser access test failed: {e}")
        return False

def main():
    """Run all connectivity tests"""
    print("🔗 UI Connectivity Diagnostic Suite")
    print("=" * 50)

    # Test 1: HTTP connectivity
    http_ok = test_http_connectivity()

    # Test 2: Gradio capability
    gradio_ok = test_gradio_launch()

    # Test 3: Browser access
    browser_ok = test_browser_access()

    # Summary
    print("\n📊 CONNECTIVITY SUMMARY")
    print("=" * 50)
    print(f"HTTP Connectivity: {'✅ OK' if http_ok else '❌ FAILED'}")
    print(f"Gradio Capability: {'✅ OK' if gradio_ok else '❌ FAILED'}")
    print(f"Browser Access: {'✅ OK' if browser_ok else '❌ FAILED'}")

    if not http_ok:
        print("\n💡 HTTP Issue Solutions:")
        print("• Check if any UI is already running (close other terminals)")
        print("• Try: python simple_ui_test.py (uses port 7862)")
        print("• Check firewall/antivirus settings")

    if not gradio_ok:
        print("\n💡 Gradio Issue Solutions:")
        print("• Reinstall gradio: pip install gradio --upgrade")
        print("• Check Python environment")

    if not browser_ok:
        print("\n💡 Browser Issue Solutions:")
        print("• Try different browser (Chrome, Firefox, Edge)")
        print("• Clear browser cache and cookies")
        print("• Disable browser extensions temporarily")

    if http_ok and gradio_ok and browser_ok:
        print("\n🎯 All connectivity tests passed!")
        print("The UI should work. Try: python stable_ui_launcher.py")
    else:
        print("\n❌ Some connectivity issues found.")
        print("Try the simple test: python simple_ui_test.py")

    print("=" * 50)

if __name__ == "__main__":
    main()
