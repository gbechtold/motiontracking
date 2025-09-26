#!/usr/bin/env python3
"""
Motion Tracking Core Module
Universal motion detection and tracking for any application
"""

import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class MotionTracker:
    """
    Universal motion tracking system using Supervision library.
    Suitable for traffic, sports, robotics, and any motion analysis.
    """

    def __init__(self,
                 tracker_type: str = "bytetrack",
                 detection_threshold: int = 500,
                 trail_length: int = 30):
        """
        Initialize motion tracker.

        Args:
            tracker_type: Type of tracker ("bytetrack" recommended)
            detection_threshold: Minimum area for object detection
            trail_length: Length of motion trail history
        """
        self.cap = None
        self.tracker_type = tracker_type
        self.detection_threshold = detection_threshold
        self.trail_length = trail_length

        # Initialize Supervision components
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.tracker = sv.ByteTrack() if tracker_type == "bytetrack" else sv.ByteTrack()

        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )

        # Tracking data
        self.track_history = defaultdict(lambda: deque(maxlen=trail_length))
        self.object_data = defaultdict(dict)
        self.statistics = {
            'total_objects': 0,
            'active_objects': 0,
            'frame_count': 0,
            'start_time': None,
            'fps': 0
        }

    def connect_source(self, source) -> bool:
        """
        Connect to video source (file, URL, or camera).

        Args:
            source: Video source (path, URL, or camera index)

        Returns:
            bool: Success status
        """
        self.cap = cv2.VideoCapture(source)

        # Optimize for network streams
        if isinstance(source, str) and ('http' in source or 'rtsp' in source):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"✅ Connected: {width}x{height} @ {fps:.1f} FPS")
            return True

        print("❌ Failed to connect to source")
        return False

    def detect_motion(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect moving objects in frame.

        Args:
            frame: Input frame

        Returns:
            sv.Detections: Detected objects
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows and noise
        _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and convert to detections
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)

            if area > self.detection_threshold:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by aspect ratio (remove noise)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5:
                    boxes.append([x, y, x+w, y+h])

        if boxes:
            boxes_array = np.array(boxes)
            return sv.Detections(
                xyxy=boxes_array,
                confidence=np.ones(len(boxes)) * 0.9,
                class_id=np.zeros(len(boxes), dtype=int)
            )

        return sv.Detections.empty()

    def update_tracking(self, detections: sv.Detections) -> sv.Detections:
        """
        Update object tracking with new detections.

        Args:
            detections: New detections

        Returns:
            sv.Detections: Tracked detections with IDs
        """
        # Update tracker
        tracked = self.tracker.update_with_detections(detections)

        # Update tracking history
        if tracked.tracker_id is not None:
            for tracker_id, bbox in zip(tracked.tracker_id, tracked.xyxy):
                # Calculate center point
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                self.track_history[tracker_id].append(center)

                # Update object data
                if tracker_id not in self.object_data:
                    self.object_data[tracker_id] = {
                        'first_seen': self.statistics['frame_count'],
                        'last_seen': self.statistics['frame_count'],
                        'positions': []
                    }
                    self.statistics['total_objects'] = max(
                        self.statistics['total_objects'],
                        tracker_id + 1
                    )

                self.object_data[tracker_id]['last_seen'] = self.statistics['frame_count']
                self.object_data[tracker_id]['positions'].append(center)

        self.statistics['active_objects'] = len(tracked)
        return tracked

    def annotate_frame(self,
                      frame: np.ndarray,
                      detections: sv.Detections,
                      show_trails: bool = True,
                      show_stats: bool = True) -> np.ndarray:
        """
        Annotate frame with tracking visualization.

        Args:
            frame: Input frame
            detections: Tracked detections
            show_trails: Show motion trails
            show_stats: Show statistics overlay

        Returns:
            np.ndarray: Annotated frame
        """
        annotated = frame.copy()

        # Draw bounding boxes
        if len(detections) > 0:
            annotated = self.box_annotator.annotate(annotated, detections)

            # Add tracker IDs
            if detections.tracker_id is not None:
                for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
                    x1, y1 = int(bbox[0]), int(bbox[1])
                    label = f"ID: {tracker_id}"
                    cv2.putText(annotated, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw motion trails
        if show_trails:
            for tracker_id, history in self.track_history.items():
                if len(history) > 1:
                    points = np.array(history, dtype=np.int32)

                    # Draw gradient trail
                    for i in range(1, len(points)):
                        alpha = i / len(points)
                        color = (
                            int(255 * (1 - alpha)),  # Red decreases
                            int(255 * alpha),         # Green increases
                            128                       # Blue constant
                        )
                        cv2.line(annotated, tuple(points[i-1]), tuple(points[i]), color, 2)

        # Add statistics overlay
        if show_stats:
            self._add_stats_overlay(annotated)

        return annotated

    def _add_stats_overlay(self, frame: np.ndarray):
        """Add statistics overlay to frame."""
        height, width = frame.shape[:2]

        # Calculate FPS
        if self.statistics['start_time']:
            elapsed = time.time() - self.statistics['start_time']
            self.statistics['fps'] = self.statistics['frame_count'] / elapsed if elapsed > 0 else 0

        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Add text
        stats = [
            f"FPS: {self.statistics['fps']:.1f}",
            f"Active: {self.statistics['active_objects']}",
            f"Total: {self.statistics['total_objects']}",
            f"Frame: {self.statistics['frame_count']}"
        ]

        y = 35
        for stat in stats:
            cv2.putText(frame, stat, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 25

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, sv.Detections]:
        """
        Process single frame through detection and tracking pipeline.

        Args:
            frame: Input frame

        Returns:
            Tuple[np.ndarray, sv.Detections]: Annotated frame and detections
        """
        self.statistics['frame_count'] += 1

        # Detect motion
        detections = self.detect_motion(frame)

        # Update tracking
        tracked = self.update_tracking(detections)

        # Annotate frame
        annotated = self.annotate_frame(frame, tracked)

        return annotated, tracked

    def run(self,
            duration: Optional[int] = None,
            save_interval: int = 0,
            output_dir: str = "output") -> Dict:
        """
        Run motion tracking loop.

        Args:
            duration: Run duration in seconds (None for unlimited)
            save_interval: Save frame every N frames (0 to disable)
            output_dir: Directory for saved frames

        Returns:
            Dict: Final statistics
        """
        if not self.cap or not self.cap.isOpened():
            print("❌ No video source connected")
            return self.statistics

        self.statistics['start_time'] = time.time()
        print("🚀 Motion tracking started! Press 'q' to quit\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Check duration
                if duration:
                    elapsed = time.time() - self.statistics['start_time']
                    if elapsed > duration:
                        break

                # Process frame
                annotated, tracked = self.process_frame(frame)

                # Save frame if requested
                if save_interval > 0 and self.statistics['frame_count'] % save_interval == 0:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{output_dir}/frame_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"💾 Saved: {filename}")

                # Display frame
                try:
                    cv2.imshow('Motion Tracker', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    pass  # Headless environment

        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")

        finally:
            self.cleanup()

        return self.statistics

    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        # Print summary
        if self.statistics['start_time']:
            elapsed = time.time() - self.statistics['start_time']
            print("\n" + "=" * 50)
            print("📊 TRACKING SUMMARY")
            print("=" * 50)
            print(f"Duration: {elapsed:.1f}s")
            print(f"Frames: {self.statistics['frame_count']}")
            print(f"Average FPS: {self.statistics['frame_count']/elapsed:.1f}")
            print(f"Total Objects: {self.statistics['total_objects']}")

    def get_statistics(self) -> Dict:
        """Get current tracking statistics."""
        return self.statistics.copy()

    def get_object_data(self, object_id: Optional[int] = None) -> Dict:
        """Get tracking data for specific object or all objects."""
        if object_id is not None:
            return self.object_data.get(object_id, {})
        return dict(self.object_data)