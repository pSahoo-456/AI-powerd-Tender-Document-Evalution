"""
Check if web servers are running on expected ports
"""

import socket
import time

def check_port(host, port, timeout=3):
    """Check if a port is open on a host"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def main():
    """Check if Streamlit servers are running"""
    print("Checking web servers...")
    
    # Give servers a moment to start
    time.sleep(2)
    
    # Check ports 8501, 8502, 8503, 8504, 8506, 8507
    ports = [8501, 8502, 8503, 8504, 8506, 8507]
    descriptions = [
        "Streamlit Demo (Original)", 
        "Main Interface (Original)", 
        "Professional Interface (Old)", 
        "Professional Interface (New)",
        "Professional Interface (Latest)",
        "Demo Interface (Latest)"
    ]
    
    for port, description in zip(ports, descriptions):
        if check_port("localhost", port):
            print(f"✓ {description} is running on port {port}")
        else:
            print(f"✗ {description} is not accessible on port {port}")
    
    print("\nWeb interfaces:")
    print("1. Demo interface (Original): http://localhost:8501")
    print("2. Main interface (Original): http://localhost:8502")
    print("3. Professional interface (Old): http://localhost:8503")
    print("4. Professional interface (New): http://localhost:8504")
    print("5. Professional interface (Latest): http://localhost:8506")
    print("6. Demo interface (Latest): http://localhost:8507")

if __name__ == "__main__":
    main()