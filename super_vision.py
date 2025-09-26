#!/usr/bin/env python3
import cv2
import numpy as np
import time
from datetime import datetime
import sys
import signal
import threading
import queue

class SuperVision:
    def __init__(self, stream_url=None):
        self.stream_url = stream_url
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.stats = {
            'frames_processed': 0,
            'start_time': None,
            'fps': 0
        }

    def connect_to_stream(self, url):
        """Connect to video stream"""
        try:
            print(f"🎥 Connecting to stream: {url}")
            self.cap = cv2.VideoCapture(url)

            if not self.cap.isOpened():
                raise Exception("Failed to open video stream")

            # Get stream properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"✅ Connected successfully!")
            print(f"📐 Resolution: {width}x{height}")
            print(f"🎬 FPS: {fps if fps > 0 else 'Unknown'}")

            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def process_frame(self, frame):
        """Process individual frame - add your vision processing here"""
        # Example: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Example: Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Example: Motion detection (simplified)
        # You can expand this with background subtraction, optical flow, etc.

        return frame, edges

    def capture_loop(self):
        """Background thread for capturing frames"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    try:
                        self.frame_queue.put(frame, timeout=0.1)
                    except queue.Full:
                        # Drop frame if queue is full
                        pass
                else:
                    print("⚠️  Failed to read frame")
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def run(self):
        """Main processing loop"""
        if not self.cap or not self.cap.isOpened():
            print("❌ No stream connected")
            return

        self.is_running = True
        self.stats['start_time'] = time.time()

        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_loop)
        capture_thread.start()

        print("\n🚀 Super Vision is running!")
        print("Press 'q' to quit, 's' to save screenshot, 'r' to show raw feed\n")

        show_raw = False

        try:
            while self.is_running:
                try:
                    frame = self.frame_queue.get(timeout=1.0)

                    # Process frame
                    processed_frame, edges = self.process_frame(frame)

                    # Update stats
                    self.stats['frames_processed'] += 1
                    elapsed = time.time() - self.stats['start_time']
                    self.stats['fps'] = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0

                    # Add info overlay
                    info_text = f"FPS: {self.stats['fps']:.1f} | Frames: {self.stats['frames_processed']}"
                    cv2.putText(processed_frame, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Display frames
                    if show_raw:
                        cv2.imshow('Super Vision - Raw Feed', frame)
                    else:
                        cv2.imshow('Super Vision - Processed', processed_frame)
                    cv2.imshow('Super Vision - Edge Detection', edges)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f'screenshot_{timestamp}.png'
                        cv2.imwrite(filename, processed_frame)
                        print(f"📸 Screenshot saved: {filename}")
                    elif key == ord('r'):
                        show_raw = not show_raw
                        if show_raw:
                            print("🎥 Showing raw feed")
                        else:
                            print("🔧 Showing processed feed")
                            cv2.destroyWindow('Super Vision - Raw Feed')

                except queue.Empty:
                    continue

        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
        finally:
            self.cleanup()
            capture_thread.join(timeout=2.0)

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        # Print final stats
        if self.stats['start_time']:
            total_time = time.time() - self.stats['start_time']
            print(f"\n📊 Session Stats:")
            print(f"  - Total frames: {self.stats['frames_processed']}")
            print(f"  - Duration: {total_time:.1f}s")
            print(f"  - Average FPS: {self.stats['fps']:.1f}")

def signal_handler(sig, frame):
    print("\n⏹️  Shutdown signal received")
    sys.exit(0)

def main():
    print("=" * 50)
    print("🔮 SUPER VISION - Video Stream POC")
    print("=" * 50)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Public test streams (replace with your preferred stream)
    test_streams = {
        '1': ('https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8', 'Mux Test Stream (HLS)'),
        '2': ('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4', 'Big Buck Bunny (MP4)'),
        '3': ('https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4', 'Sample Video (MP4)'),
        '4': ('custom', 'Enter custom URL')
    }

    print("\nAvailable test streams:")
    for key, (url, name) in test_streams.items():
        if url != 'custom':
            print(f"  {key}. {name}")
        else:
            print(f"  {key}. {name}")

    choice = input("\nSelect stream (1-4): ").strip()

    if choice in test_streams:
        stream_url, name = test_streams[choice]
        if stream_url == 'custom':
            stream_url = input("Enter stream URL: ").strip()
    else:
        print("Invalid choice, using default stream")
        stream_url = test_streams['1'][0]

    # Create Super Vision instance
    vision = SuperVision()

    # Connect to stream
    if vision.connect_to_stream(stream_url):
        # Run processing
        vision.run()
    else:
        print("Failed to connect to stream")
        sys.exit(1)

if __name__ == "__main__":
    main()