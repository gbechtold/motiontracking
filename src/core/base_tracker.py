#!/usr/bin/env python3
"""
Super Vision with Supervision Library - Simple Version
Object Detection and Tracking without YOLO
"""

import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
import time
import sys

class SimpleTracker:
    def __init__(self):
        self.cap = None

        # Initialize supervision components
        self.box_annotator = sv.BoxAnnotator(
            thickness=2
        )

        # Motion detector
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True
        )

        # Tracker
        self.tracker = sv.ByteTrack()

        # Track history
        self.track_history = defaultdict(lambda: [])

        # Stats
        self.frame_count = 0
        self.start_time = None

    def connect_to_stream(self, url):
        """Connect to video stream"""
        print(f"\n🎥 Connecting to stream: {url}")
        self.cap = cv2.VideoCapture(url)

        if not self.cap.isOpened():
            print("❌ Failed to open stream")
            return False

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"✅ Connected!")
        print(f"📐 Resolution: {width}x{height}")
        print(f"🎬 FPS: {fps:.1f}")
        print("-" * 50)

        return True

    def detect_motion_objects(self, frame):
        """Detect moving objects using background subtraction"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Remove shadows and noise
        _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Convert to detections
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, x+w, y+h])

        if boxes:
            boxes = np.array(boxes)
            # Create supervision Detections
            detections = sv.Detections(
                xyxy=boxes,
                confidence=np.ones(len(boxes)) * 0.8,
                class_id=np.zeros(len(boxes), dtype=int)
            )
            return detections

        return sv.Detections.empty()

    def process_frame(self, frame):
        """Process frame with detection and tracking"""
        # Detect objects
        detections = self.detect_motion_objects(frame)

        # Track objects
        tracked_detections = self.tracker.update_with_detections(detections)

        # Update tracking history
        if tracked_detections.tracker_id is not None:
            for tracker_id, bbox in zip(tracked_detections.tracker_id, tracked_detections.xyxy):
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                self.track_history[tracker_id].append(center)

                # Keep only last 30 positions
                if len(self.track_history[tracker_id]) > 30:
                    self.track_history[tracker_id].pop(0)

        return tracked_detections

    def annotate_frame(self, frame, detections):
        """Add visual annotations to frame"""
        annotated = frame.copy()

        # Draw bounding boxes
        if len(detections) > 0:
            annotated = self.box_annotator.annotate(
                scene=annotated,
                detections=detections
            )

            # Add labels if tracker IDs exist
            if detections.tracker_id is not None:
                for i, (tracker_id, bbox) in enumerate(zip(detections.tracker_id, detections.xyxy)):
                    x1, y1, x2, y2 = bbox.astype(int)
                    label = f"ID: {tracker_id}"

                    # Draw label
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

        # Draw tracking trails
        for tracker_id, history in self.track_history.items():
            if len(history) > 1:
                points = np.array(history, dtype=np.int32)
                cv2.polylines(annotated, [points], False, (0, 255, 255), 2)

                # Draw dots at each point
                for point in points[-5:]:  # Last 5 points
                    cv2.circle(annotated, tuple(point.astype(int)), 3, (255, 255, 0), -1)

        return annotated

    def run(self, duration=30):
        """Main tracking loop"""
        if not self.cap:
            print("❌ No stream connected")
            return

        self.start_time = time.time()
        print("\n🚀 Tracking started!\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1

                # Check duration
                elapsed = time.time() - self.start_time
                if duration and elapsed > duration:
                    print(f"\n⏱️  Duration limit ({duration}s) reached")
                    break

                # Process frame
                detections = self.process_frame(frame)

                # Annotate frame
                annotated_frame = self.annotate_frame(frame, detections)

                # Add stats overlay
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                stats_text = [
                    f"FPS: {fps:.1f}",
                    f"Frame: {self.frame_count}",
                    f"Objects: {len(detections)}",
                    f"Tracked: {len(self.track_history)}"
                ]

                y_offset = 30
                for text in stats_text:
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    y_offset += 30

                # Save frame periodically
                if self.frame_count % 100 == 0:
                    filename = f"tracked_frame_{self.frame_count:06d}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"💾 Saved: {filename}")

                # Print stats
                if self.frame_count % 10 == 0:
                    print(f"\r📊 Frame: {self.frame_count} | FPS: {fps:.1f} | Objects: {len(detections)} | Tracked IDs: {len(self.track_history)}", end="")

        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 50)
        print("📊 TRACKING SUMMARY")
        print("=" * 50)

        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"Duration: {elapsed:.1f}s")
        print(f"Total Frames: {self.frame_count}")
        print(f"Average FPS: {self.frame_count/elapsed:.1f}" if elapsed > 0 else "N/A")
        print(f"Total Objects Tracked: {len(self.track_history)}")

        if self.track_history:
            print("\n🎯 Top Moving Objects:")
            sorted_tracks = sorted(
                self.track_history.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:5]

            for track_id, history in sorted_tracks:
                print(f"  • Object #{track_id}: {len(history)} positions tracked")

def main():
    print("=" * 50)
    print("🔮 SUPER VISION - Motion Tracking")
    print("=" * 50)

    # Default to Big Buck Bunny
    stream_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

    print(f"\n📹 Using stream: Big Buck Bunny")
    print("⏱️  Running for 30 seconds...")

    # Create and run tracker
    tracker = SimpleTracker()

    if tracker.connect_to_stream(stream_url):
        tracker.run(duration=30)
    else:
        print("Failed to connect")
        sys.exit(1)

if __name__ == "__main__":
    main()