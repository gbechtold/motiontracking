#!/usr/bin/env python3
"""
Super Vision - Real Traffic Camera Stream Analyzer
Tests multiple real traffic camera sources
"""

import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
import time
import sys
from datetime import datetime
import subprocess

class RealTrafficMonitor:
    def __init__(self):
        self.cap = None
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.tracker = sv.ByteTrack()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=300
        )

        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.vehicle_counts = defaultdict(int)
        self.frame_count = 0
        self.start_time = None

    def test_stream_connection(self, url, name):
        """Test if stream is accessible"""
        print(f"\n🔍 Testing: {name}")
        if isinstance(url, str):
            print(f"   URL: {url[:60]}..." if len(url) > 60 else f"   URL: {url}")
        else:
            print(f"   URL: Local camera {url}")

        try:
            test_cap = cv2.VideoCapture(url)
            test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Try to read a frame
            for _ in range(3):
                ret, frame = test_cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    print(f"   ✅ SUCCESS! Resolution: {w}x{h}")
                    test_cap.release()
                    return True
                time.sleep(0.5)

            test_cap.release()
            print(f"   ❌ Failed - No frames received")
            return False

        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")
            return False

    def connect_to_stream(self, url):
        """Connect to selected stream"""
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # For MJPEG streams
        if '.mjpg' in url or 'mjpeg' in url.lower():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        if not self.cap.isOpened():
            return False

        return True

    def detect_vehicles(self, frame):
        """Detect moving vehicles"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows and noise
        _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio
                if 0.3 < w/h < 4:
                    boxes.append([x, y, x+w, y+h])

        if boxes:
            boxes = np.array(boxes)
            return sv.Detections(
                xyxy=boxes,
                confidence=np.ones(len(boxes)) * 0.9,
                class_id=np.zeros(len(boxes), dtype=int)
            )

        return sv.Detections.empty()

    def run_analysis(self, url, name, duration=30):
        """Analyze traffic stream"""
        print(f"\n🚦 ANALYZING: {name}")
        print("=" * 60)

        if not self.connect_to_stream(url):
            print("❌ Connection failed")
            return False

        self.start_time = time.time()
        self.frame_count = 0
        total_vehicles = 0

        print("🎬 Stream connected! Analyzing traffic...")
        print("Press Ctrl+C to stop\n")

        try:
            while time.time() - self.start_time < duration:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                self.frame_count += 1

                # Detect and track vehicles
                detections = self.detect_vehicles(frame)
                tracked = self.tracker.update_with_detections(detections)

                # Update tracking history
                if tracked.tracker_id is not None:
                    for tracker_id, bbox in zip(tracked.tracker_id, tracked.xyxy):
                        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                        self.track_history[tracker_id].append(center)

                        # Count new vehicles
                        if tracker_id > total_vehicles:
                            total_vehicles = tracker_id

                # Annotate frame
                annotated = frame.copy()
                if len(tracked) > 0:
                    annotated = self.box_annotator.annotate(annotated, tracked)

                    # Draw trails
                    for tracker_id, history in self.track_history.items():
                        if len(history) > 1:
                            points = np.array(history, dtype=np.int32)
                            cv2.polylines(annotated, [points], False, (0, 255, 0), 2)

                # Add overlay
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0

                cv2.rectangle(annotated, (10, 10), (350, 120), (0, 0, 0), -1)
                cv2.putText(annotated, f"REAL TRAFFIC: {name[:30]}", (15, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(annotated, f"Time: {datetime.now().strftime('%H:%M:%S')}", (15, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated, f"Active: {len(tracked)} | Total: {total_vehicles}", (15, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated, f"FPS: {fps:.1f}", (15, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Save snapshot every 5 seconds
                if self.frame_count % (int(fps * 5) + 1) == 0 and fps > 0:
                    filename = f"real_traffic_{self.frame_count:06d}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"💾 Snapshot saved: {filename} | Active: {len(tracked)} | Total: {total_vehicles}")

                # Display if possible
                try:
                    cv2.imshow(f'Real Traffic - {name}', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    pass

        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

            elapsed = time.time() - self.start_time
            print(f"\n📊 RESULTS for {name}:")
            print(f"   Duration: {elapsed:.1f}s")
            print(f"   Frames: {self.frame_count}")
            print(f"   FPS: {self.frame_count/elapsed:.1f}")
            print(f"   Total Vehicles Detected: {total_vehicles}")

        return True

def main():
    print("=" * 70)
    print("🚦 SUPER VISION - REAL TRAFFIC CAMERA ANALYZER")
    print("=" * 70)

    # Real traffic camera sources
    traffic_streams = [
        {
            'name': 'Times Square, NYC',
            'url': 'https://webcams.nyctmc.org/api/cameras/473/image/refresh',
            'type': 'image_refresh'
        },
        {
            'name': 'Jackson Hole Town Square',
            'url': 'https://www.youtube.com/watch?v=1EiC9bvVGnk',
            'type': 'youtube',
            'note': 'Live stream - requires yt-dlp'
        },
        {
            'name': 'Dublin Ireland - O\'Connell Street',
            'url': 'https://www.youtube.com/watch?v=cPR6fHTKKPU',
            'type': 'youtube',
            'note': 'Live traffic cam'
        },
        {
            'name': 'Tokyo - Shibuya Crossing',
            'url': 'https://www.youtube.com/watch?v=OrFJAQgXPHo',
            'type': 'youtube',
            'note': 'Famous crossing'
        },
        {
            'name': 'Abbey Road Crossing, London',
            'url': 'https://www.youtube.com/watch?v=KGuCGd726RA',
            'type': 'youtube',
            'note': 'Beatles crossing'
        },
        {
            'name': 'Test Local Webcam',
            'url': '0',
            'type': 'local',
            'note': 'Your computer camera'
        }
    ]

    monitor = RealTrafficMonitor()

    print("\n🔍 Testing available traffic cameras...")
    print("-" * 60)

    working_streams = []

    # Test non-YouTube streams
    for i, stream in enumerate(traffic_streams, 1):
        if stream['type'] != 'youtube':
            if stream['type'] == 'local':
                url = 0
            else:
                url = stream['url']

            if monitor.test_stream_connection(url, stream['name']):
                working_streams.append(stream)

    # For YouTube streams, try yt-dlp
    print("\n📹 Checking YouTube streams...")
    try:
        result = subprocess.run(['which', 'yt-dlp'], capture_output=True)
        if result.returncode == 0:
            print("✅ yt-dlp is available for YouTube streams")

            # Try to get direct URL from first YouTube stream
            for stream in traffic_streams:
                if stream['type'] == 'youtube':
                    print(f"\n🎥 Attempting: {stream['name']}")
                    try:
                        cmd = ['yt-dlp', '-f', 'best[height<=720]', '-g', '--no-warnings', stream['url']]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                        if result.returncode == 0:
                            direct_url = result.stdout.strip()
                            stream['direct_url'] = direct_url

                            if monitor.test_stream_connection(direct_url, stream['name']):
                                working_streams.append(stream)
                                break  # Found one working stream
                    except Exception as e:
                        print(f"   ⚠️ Could not extract: {str(e)[:30]}")
        else:
            print("⚠️ yt-dlp not found - YouTube streams unavailable")
            print("   Install with: pip install yt-dlp")
    except:
        pass

    if not working_streams:
        print("\n⚠️ No working streams found. Using test video instead.")
        working_streams.append({
            'name': 'Test Video (Demo)',
            'url': 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
            'type': 'demo'
        })

    print("\n" + "=" * 60)
    print(f"✅ Found {len(working_streams)} working stream(s)")

    # Select best available stream
    selected = working_streams[0]

    if 'direct_url' in selected:
        url = selected['direct_url']
    elif selected['type'] == 'local':
        url = 0
    else:
        url = selected['url']

    print(f"\n🎯 Selected: {selected['name']}")
    print("Starting 30-second traffic analysis...")

    # Run analysis
    monitor.run_analysis(url, selected['name'], duration=30)

if __name__ == "__main__":
    main()