#!/usr/bin/env python3
"""Quick test script to verify stream connection"""

import cv2
import sys

def test_stream(url):
    print(f"Testing stream: {url}")

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("❌ Failed to open stream")
        return False

    print("✅ Stream opened successfully")

    # Read a few frames
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"  Frame {i+1}: {frame.shape}")
        else:
            print(f"  Frame {i+1}: Failed to read")

    cap.release()
    return True

if __name__ == "__main__":
    # Test with a reliable public stream
    test_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

    if len(sys.argv) > 1:
        test_url = sys.argv[1]

    test_stream(test_url)