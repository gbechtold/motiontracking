#!/usr/bin/env python3
import cv2
import numpy as np
import time
from datetime import datetime
import sys
import signal
import threading
import queue
import os

class SuperVisionHeadless:
    def __init__(self, stream_url=None, save_frames=False):
        self.stream_url = stream_url
        self.cap = None
        self.is_running = False
        self.save_frames = save_frames
        self.frame_queue = queue.Queue(maxsize=10)
        self.stats = {
            'frames_processed': 0,
            'start_time': None,
            'fps': 0,
            'motion_detected': False,
            'last_motion_time': None,
            'objects_detected': 0
        }
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.output_dir = "output"

        if save_frames and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def connect_to_stream(self, url):
        """Connect to video stream"""
        try:
            print(f"\n🎥 Connecting to stream: {url}")
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
            print("-" * 50)

            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def detect_motion(self, frame):
        """Detect motion in frame using background subtraction"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Apply threshold
        _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append({
                    'x': x, 'y': y,
                    'width': w, 'height': h,
                    'area': area
                })

        return len(motion_areas) > 0, motion_areas

    def process_frame(self, frame):
        """Process individual frame with tracking"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect motion
        motion_detected, motion_areas = self.detect_motion(frame)

        if motion_detected:
            self.stats['motion_detected'] = True
            self.stats['last_motion_time'] = time.time()
            self.stats['objects_detected'] = len(motion_areas)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_count = cv2.countNonZero(edges)

        # Simple blob detection for objects
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)

        analysis = {
            'motion_detected': motion_detected,
            'motion_areas': motion_areas,
            'edge_density': edge_count / (frame.shape[0] * frame.shape[1]),
            'blob_count': len(keypoints),
            'brightness': np.mean(gray)
        }

        return analysis

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

    def run(self, duration=None):
        """Main processing loop (headless)"""
        if not self.cap or not self.cap.isOpened():
            print("❌ No stream connected")
            return

        self.is_running = True
        self.stats['start_time'] = time.time()

        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_loop)
        capture_thread.start()

        print("\n🚀 Super Vision Headless is running!")
        print("📊 Live tracking data:\n")

        last_save_time = time.time()
        frame_save_interval = 5  # Save frame every 5 seconds

        try:
            while self.is_running:
                # Check duration limit
                if duration and (time.time() - self.stats['start_time']) > duration:
                    print(f"\n⏱️  Duration limit ({duration}s) reached")
                    break

                try:
                    frame = self.frame_queue.get(timeout=1.0)

                    # Process frame
                    analysis = self.process_frame(frame)

                    # Update stats
                    self.stats['frames_processed'] += 1
                    elapsed = time.time() - self.stats['start_time']
                    self.stats['fps'] = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0

                    # Save frame periodically
                    if self.save_frames and (time.time() - last_save_time) > frame_save_interval:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = os.path.join(self.output_dir, f'frame_{timestamp}.jpg')
                        cv2.imwrite(filename, frame)
                        last_save_time = time.time()

                    # Print live stats every 10 frames
                    if self.stats['frames_processed'] % 10 == 0:
                        self.print_stats(analysis)

                except queue.Empty:
                    continue

        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
        finally:
            self.cleanup()
            capture_thread.join(timeout=2.0)

    def print_stats(self, analysis):
        """Print live tracking statistics"""
        # Clear previous lines (simple approach for terminal)
        print("\033[2K\033[F" * 8, end='')  # Move up and clear lines

        print(f"📊 LIVE TRACKING DATA")
        print(f"├─ FPS: {self.stats['fps']:.1f}")
        print(f"├─ Frames Processed: {self.stats['frames_processed']}")
        print(f"├─ Motion: {'🔴 DETECTED' if analysis['motion_detected'] else '⚪ None'}")
        print(f"├─ Moving Objects: {len(analysis['motion_areas'])}")
        print(f"├─ Edge Density: {analysis['edge_density']:.3f}")
        print(f"├─ Detected Blobs: {analysis['blob_count']}")
        print(f"└─ Brightness: {analysis['brightness']:.1f}/255")

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()

        # Print final stats
        print("\n" + "=" * 50)
        print("📊 FINAL SESSION STATISTICS")
        print("=" * 50)

        if self.stats['start_time']:
            total_time = time.time() - self.stats['start_time']
            print(f"Duration: {total_time:.1f} seconds")
            print(f"Total Frames: {self.stats['frames_processed']}")
            print(f"Average FPS: {self.stats['fps']:.1f}")

            if self.stats['last_motion_time']:
                motion_ago = time.time() - self.stats['last_motion_time']
                print(f"Last Motion: {motion_ago:.1f}s ago")

            if self.save_frames:
                print(f"Frames saved to: {self.output_dir}/")

def signal_handler(sig, frame):
    print("\n⏹️  Shutdown signal received")
    sys.exit(0)

def main():
    print("=" * 50)
    print("🔮 SUPER VISION - Headless Tracking Mode")
    print("=" * 50)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Test streams
    test_streams = {
        '1': 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
        '2': 'https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8',
        '3': 'https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4'
    }

    print("\n🎬 Available test streams:")
    print("1. Big Buck Bunny (MP4)")
    print("2. Mux Test Stream (HLS)")
    print("3. Sample Video (MP4)")
    print("4. Custom URL")

    choice = input("\nSelect stream (1-4): ").strip()

    if choice == '4':
        stream_url = input("Enter stream URL: ").strip()
    elif choice in test_streams:
        stream_url = test_streams[choice]
    else:
        print("Using default stream...")
        stream_url = test_streams['1']

    # Ask about saving frames
    save_frames = input("\n💾 Save frames to disk? (y/n): ").strip().lower() == 'y'

    # Ask about duration
    duration = None
    duration_input = input("\n⏱️  Run duration in seconds (press Enter for unlimited): ").strip()
    if duration_input:
        try:
            duration = int(duration_input)
        except ValueError:
            pass

    # Create Super Vision instance
    vision = SuperVisionHeadless(save_frames=save_frames)

    # Connect to stream
    if vision.connect_to_stream(stream_url):
        # Run processing
        vision.run(duration=duration)
    else:
        print("Failed to connect to stream")
        sys.exit(1)

if __name__ == "__main__":
    main()