#!/usr/bin/env python3
"""
Fixed UI launcher with better server management and error handling
"""

import sys
import os
import time
import signal
import traceback
import threading
import webbrowser

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n👋 Received interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def check_server_readiness(port, timeout=10):
    """Check if server is ready and responding"""
    import socket
    import time

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result == 0:
                return True
        except:
            pass
        time.sleep(0.5)

    return False

def launch_ui_with_monitoring():
    """Launch the UI with monitoring and better error handling"""
    print("🚀 Starting Advanced Signature Detection UI with monitoring...")
    print("=" * 70)

    try:
        from advanced_gradio_ui import AdvancedSignatureUI

        print("📦 Initializing UI...")
        ui = AdvancedSignatureUI()

        print("🎨 Creating interface...")
        interface = ui.create_interface()

        print("🌐 Server configuration:")
        print("   - Host: 127.0.0.1 (localhost only)")
        print("   - Port: 7860")
        print("   - Share: Disabled")
        print("   - Debug: Enabled")
        print()

        # Launch interface in a separate thread so we can monitor it
        def launch_server():
            try:
                interface.launch(
                    server_name="127.0.0.1",  # Only localhost for security
                    server_port=7860,
                    share=False,
                    show_error=True,
                    debug=True,
                    quiet=False,
                    prevent_thread_lock=True  # Important for monitoring
                )
            except Exception as e:
                print(f"❌ Server launch error: {e}")

        print("🔄 Starting server thread...")
        server_thread = threading.Thread(target=launch_server, daemon=True)
        server_thread.start()

        # Wait for server to be ready
        print("⏳ Waiting for server to start...")
        if check_server_readiness(7860, timeout=15):
            print("✅ Server is ready and responding!")

            # Try to open browser
            url = "http://127.0.0.1:7860"
            print(f"🌐 Opening browser to: {url}")

            try:
                # Give a moment for the server to fully initialize
                time.sleep(2)
                webbrowser.open(url)
                print("✅ Browser opened successfully!")
            except Exception as e:
                print(f"⚠️  Could not open browser automatically: {e}")
                print(f"   Please manually open: {url}")

            print("\n" + "=" * 70)
            print("🎯 UI is running successfully!")
            print("📋 Instructions:")
            print("• Keep this terminal window open")
            print("• Use Ctrl+C to stop the server")
            print("• The UI should be accessible in your browser")
            print("=" * 70)

            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 Shutting down server...")
                # Note: Gradio should handle cleanup automatically

        else:
            print("❌ Server failed to start within 15 seconds")
            print("💡 Possible causes:")
            print("• Port 7860 may be in use by another application")
            print("• Firewall blocking the connection")
            print("• Insufficient permissions")

            # Try alternative port
            print("\n🔄 Trying alternative port 7861...")
            try:
                interface.launch(
                    server_name="127.0.0.1",
                    server_port=7861,
                    share=False,
                    show_error=True,
                    debug=True
                )
            except Exception as e:
                print(f"❌ Alternative port also failed: {e}")

    except KeyboardInterrupt:
        print("\n👋 UI stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching UI: {e}")
        print("📄 Full error traceback:")
        traceback.print_exc()

        print("\n🔧 Troubleshooting:")
        print("1. Check if Python and required packages are installed:")
        print("   pip install torch torchvision transformers accelerate gradio ultralytics")
        print("2. Try the simple test UI:")
        print("   python simple_ui_test.py")
        print("3. Check firewall/antivirus settings")
        print("4. Try running as administrator")

def main():
    """Main function"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🎯 Advanced Signature Detection UI - Fixed Launcher")
    print("🔧 With monitoring and better error handling")
    print("=" * 70)

    # Check if we're in the right directory
    if not os.path.exists("weights/yolov8s.pt"):
        print("❌ Error: yolov8s.pt not found in weights/ directory")
        print("   Make sure you're running from the correct directory")
        sys.exit(1)

    print("✅ Found yolov8s.pt - proceeding...")
    print()

    launch_ui_with_monitoring()

if __name__ == "__main__":
    main()
