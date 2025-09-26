#!/usr/bin/env python3
"""
Super Vision with Supervision Library
Advanced Object Detection and Tracking
"""

import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
import time
import sys
import os

class SuperVisionTracker:
    def __init__(self, stream_url=None):
        self.stream_url = stream_url
        self.cap = None

        # Initialize supervision components
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )

        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=5
        )

        # Initialize tracker
        self.tracker = sv.ByteTrack()

        # Track history
        self.track_history = defaultdict(lambda: [])
        self.track_info = {}

        # Stats
        self.frame_count = 0
        self.start_time = None

        # Try to use YOLO if available, otherwise use simple detection
        self.model = None
        try:
            from ultralytics import YOLO
            # Download and use YOLOv8n (nano) model - smallest and fastest
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"⚠️  YOLO not available, using fallback detection: {e}")

    def connect_to_stream(self, url):
        """Connect to video stream"""
        try:
            print(f"\n🎥 Connecting to stream: {url}")
            self.cap = cv2.VideoCapture(url)

            if not self.cap.isOpened():
                raise Exception("Failed to open video stream")

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

    def detect_objects_fallback(self, frame):
        """Fallback detection using traditional CV methods"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Simple blob detection
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.minArea = 500
        detector_params.maxArea = 50000
        detector = cv2.SimpleBlobDetector_create(detector_params)

        keypoints = detector.detect(gray)

        # Convert keypoints to detections
        detections = []
        for kp in keypoints:
            x, y = kp.pt
            size = kp.size * 2  # Approximate bounding box size
            x1 = max(0, int(x - size/2))
            y1 = max(0, int(y - size/2))
            x2 = min(frame.shape[1], int(x + size/2))
            y2 = min(frame.shape[0], int(y + size/2))

            detections.append([x1, y1, x2, y2, 0.5, 0])  # bbox, confidence, class

        if detections:
            detections = np.array(detections)
            return sv.Detections(
                xyxy=detections[:, :4],
                confidence=detections[:, 4],
                class_id=detections[:, 5].astype(int)
            )
        else:
            return sv.Detections.empty()

    def detect_objects_yolo(self, frame):
        """Detect objects using YOLO"""
        results = self.model(frame, verbose=False)[0]

        # Convert YOLO results to supervision Detections
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.empty((0, 4)),
            confidence=results.boxes.conf.cpu().numpy() if results.boxes is not None else np.empty(0),
            class_id=results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else np.empty(0, dtype=int)
        )

        return detections

    def process_frame(self, frame):
        """Process frame with object detection and tracking"""
        # Detect objects
        if self.model:
            detections = self.detect_objects_yolo(frame)
        else:
            detections = self.detect_objects_fallback(frame)

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

                # Store track info
                if tracker_id not in self.track_info:
                    self.track_info[tracker_id] = {
                        'first_seen': self.frame_count,
                        'last_seen': self.frame_count,
                        'total_frames': 1
                    }
                else:
                    self.track_info[tracker_id]['last_seen'] = self.frame_count
                    self.track_info[tracker_id]['total_frames'] += 1

        return tracked_detections

    def annotate_frame(self, frame, detections):
        """Add annotations to frame"""
        # Generate labels
        labels = []
        if self.model and detections.class_id is not None:
            for tracker_id, class_id, confidence in zip(
                detections.tracker_id if detections.tracker_id is not None else range(len(detections)),
                detections.class_id,
                detections.confidence
            ):
                class_name = self.model.names.get(class_id, f"Class {class_id}") if self.model else f"Object"
                label = f"#{tracker_id} {class_name} {confidence:.2f}"
                labels.append(label)
        else:
            labels = [f"Object #{i}" for i in range(len(detections))]

        # Annotate with boxes
        annotated_frame = frame.copy()
        if len(detections) > 0:
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

            if labels:
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )

        # Draw tracking trails
        for tracker_id, history in self.track_history.items():
            if len(history) > 1:
                points = np.array(history, dtype=np.int32)
                cv2.polylines(annotated_frame, [points], False, (0, 255, 0), 2)

        return annotated_frame

    def print_stats(self, detections):
        """Print tracking statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        # Clear previous lines
        print("\033[2K\033[F" * 10, end='')

        print("=" * 50)
        print("📊 OBJECT TRACKING STATISTICS")
        print("-" * 50)
        print(f"FPS: {fps:.1f}")
        print(f"Frames: {self.frame_count}")
        print(f"Active Objects: {len(detections)}")
        print(f"Total Tracked: {len(self.track_info)}")

        if self.model and len(detections) > 0 and detections.class_id is not None:
            # Count objects by class
            unique_classes, counts = np.unique(detections.class_id, return_counts=True)
            print("\n🎯 Detected Objects:")
            for class_id, count in zip(unique_classes, counts):
                class_name = self.model.names.get(class_id, f"Class {class_id}")
                print(f"  • {class_name}: {count}")

    def run(self, save_video=False, duration=None):
        """Main processing loop"""
        if not self.cap or not self.cap.isOpened():
            print("❌ No stream connected")
            return

        self.start_time = time.time()

        # Setup video writer if saving
        writer = None
        if save_video:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30

            output_file = f"tracked_output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            print(f"💾 Saving to: {output_file}")

        print("\n🚀 Tracking started! Press Ctrl+C to stop.\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Failed to read frame")
                    break

                self.frame_count += 1

                # Check duration
                if duration and (time.time() - self.start_time) > duration:
                    print(f"\n⏱️  Duration limit ({duration}s) reached")
                    break

                # Process frame
                detections = self.process_frame(frame)

                # Annotate frame
                annotated_frame = self.annotate_frame(frame, detections)

                # Save frame if recording
                if writer:
                    writer.write(annotated_frame)

                # Print stats every 10 frames
                if self.frame_count % 10 == 0:
                    self.print_stats(detections)

                # Try to display (might not work in headless)
                try:
                    cv2.imshow('Super Vision Tracker', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    pass  # Headless environment

        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping...")

        finally:
            self.cleanup(writer)

    def cleanup(self, writer=None):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Print final statistics
        print("\n" + "=" * 50)
        print("📊 FINAL TRACKING REPORT")
        print("=" * 50)

        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"Duration: {elapsed:.1f}s")
        print(f"Total Frames: {self.frame_count}")
        print(f"Average FPS: {self.frame_count/elapsed:.1f}" if elapsed > 0 else "N/A")
        print(f"Total Objects Tracked: {len(self.track_info)}")

        if self.track_info:
            print("\n🎯 Object Persistence:")
            for track_id, info in sorted(self.track_info.items())[:10]:  # Show top 10
                duration = (info['last_seen'] - info['first_seen']) / (self.frame_count/elapsed) if elapsed > 0 else 0
                print(f"  • Object #{track_id}: {info['total_frames']} frames ({duration:.1f}s)")

def main():
    print("=" * 50)
    print("🔮 SUPER VISION - Object Tracking with Supervision")
    print("=" * 50)

    # Test streams
    test_streams = {
        '1': 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
        '2': 'https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8',
        '3': 'custom'
    }

    print("\n🎬 Select stream:")
    print("1. Big Buck Bunny (MP4)")
    print("2. Mux Test Stream (HLS)")
    print("3. Custom URL")

    choice = input("\nChoice (1-3): ").strip()

    if choice == '3':
        stream_url = input("Enter stream URL: ").strip()
    elif choice in test_streams:
        stream_url = test_streams[choice]
    else:
        stream_url = test_streams['1']

    # Options
    save = input("\n💾 Save tracked video? (y/n): ").strip().lower() == 'y'
    duration_str = input("⏱️  Duration in seconds (Enter for unlimited): ").strip()
    duration = int(duration_str) if duration_str else None

    # Create tracker
    tracker = SuperVisionTracker()

    # Connect and run
    if tracker.connect_to_stream(stream_url):
        tracker.run(save_video=save, duration=duration)
    else:
        print("Failed to connect to stream")
        sys.exit(1)

if __name__ == "__main__":
    main()