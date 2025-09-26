#!/usr/bin/env python3
"""
Super Vision - Traffic Monitoring with YouTube Live Streams
Uses public YouTube traffic cameras
"""

import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
import time
import sys
import subprocess
import os
from datetime import datetime

class YouTubeTrafficTracker:
    def __init__(self):
        self.cap = None
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.tracker = sv.ByteTrack()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        self.track_history = defaultdict(lambda: [])
        self.vehicle_counts = defaultdict(int)
        self.frame_count = 0
        self.start_time = None

    def get_youtube_stream_url(self, youtube_url):
        """Extract direct stream URL from YouTube (requires yt-dlp)"""
        try:
            print("📥 Extracting stream URL from YouTube...")

            # Check if yt-dlp is installed
            result = subprocess.run(
                ['which', 'yt-dlp'],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print("⚠️  yt-dlp not found. Installing...")
                subprocess.run(['pip', 'install', 'yt-dlp'], check=True)

            # Get stream URL
            cmd = [
                'yt-dlp',
                '-f', 'best[height<=720]',
                '-g',
                '--no-warnings',
                youtube_url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                stream_url = result.stdout.strip()
                return stream_url
            else:
                print(f"❌ Failed to extract stream: {result.stderr}")
                return None

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def connect_to_stream(self, url):
        """Connect to video stream"""
        print(f"\n🚦 Connecting to traffic camera...")

        # If it's a YouTube URL, extract the stream
        if 'youtube.com' in url or 'youtu.be' in url:
            stream_url = self.get_youtube_stream_url(url)
            if not stream_url:
                return False
            url = stream_url

        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("❌ Failed to open stream")
            return False

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"✅ Connected!")
        print(f"📐 Resolution: {width}x{height}")
        print("-" * 60)

        return True

    def detect_vehicles(self, frame):
        """Detect vehicles using motion detection"""
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, x+w, y+h])

        if boxes:
            boxes = np.array(boxes)
            return sv.Detections(
                xyxy=boxes,
                confidence=np.ones(len(boxes)) * 0.8,
                class_id=np.zeros(len(boxes), dtype=int)
            )

        return sv.Detections.empty()

    def run(self, duration=60):
        """Main processing loop"""
        if not self.cap:
            print("❌ No stream connected")
            return

        self.start_time = time.time()
        print("\n🚦 TRAFFIC MONITORING ACTIVE")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                self.frame_count += 1
                elapsed = time.time() - self.start_time

                if duration and elapsed > duration:
                    break

                # Process frame
                detections = self.detect_vehicles(frame)
                tracked = self.tracker.update_with_detections(detections)

                # Update statistics
                if tracked.tracker_id is not None:
                    for tracker_id, bbox in zip(tracked.tracker_id, tracked.xyxy):
                        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                        self.track_history[tracker_id].append(center)

                        if len(self.track_history[tracker_id]) > 30:
                            self.track_history[tracker_id].pop(0)

                # Annotate frame
                annotated = frame.copy()
                if len(tracked) > 0:
                    annotated = self.box_annotator.annotate(annotated, tracked)

                    # Add tracking trails
                    for tracker_id, history in self.track_history.items():
                        if len(history) > 1:
                            points = np.array(history, dtype=np.int32)
                            cv2.polylines(annotated, [points], False, (0, 255, 0), 2)

                # Add stats overlay
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Vehicles: {len(tracked)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Total Tracked: {len(self.track_history)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Save frame periodically
                if self.frame_count % 200 == 0:
                    filename = f"traffic_{self.frame_count:06d}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"💾 Saved: {filename}")

                # Print stats
                if self.frame_count % 30 == 0:
                    print(f"📊 Frame: {self.frame_count} | FPS: {fps:.1f} | "
                          f"Active: {len(tracked)} | Total: {len(self.track_history)}")

        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("📊 TRAFFIC MONITORING SUMMARY")
        print("=" * 60)

        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        print(f"Duration: {elapsed:.1f}s")
        print(f"Total Frames: {self.frame_count}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Total Vehicles Tracked: {len(self.track_history)}")

def main():
    print("=" * 60)
    print("🚦 SUPER VISION - TRAFFIC MONITORING")
    print("=" * 60)

    # Use a simple test video for demo
    print("\n📹 Using test traffic video for demonstration")
    print("(Live streams require yt-dlp and may be geo-restricted)")

    # Test with Big Buck Bunny (will show movement patterns)
    stream_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

    tracker = YouTubeTrafficTracker()

    if tracker.connect_to_stream(stream_url):
        tracker.run(duration=30)
    else:
        print("Failed to connect")

if __name__ == "__main__":
    main()