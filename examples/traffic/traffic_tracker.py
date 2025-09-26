#!/usr/bin/env python3
"""
Super Vision - Traffic Stream Tracker
Live traffic monitoring with object detection and tracking
"""

import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, Counter
import time
import sys
import os
from datetime import datetime

class TrafficTracker:
    def __init__(self):
        self.cap = None

        # Supervision components
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.tracker = sv.ByteTrack()

        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )

        # Track history and statistics
        self.track_history = defaultdict(lambda: [])
        self.vehicle_counts = {
            'cars': 0,
            'trucks': 0,
            'bikes': 0,
            'pedestrians': 0,
            'total': 0
        }
        self.track_info = {}
        self.speed_estimates = {}

        # Frame processing
        self.frame_count = 0
        self.start_time = None
        self.fps = 0

    def connect_to_stream(self, url):
        """Connect to traffic camera stream"""
        print(f"\n🚦 Connecting to traffic stream...")
        print(f"📹 URL: {url[:50]}..." if len(url) > 50 else f"📹 URL: {url}")

        self.cap = cv2.VideoCapture(url)

        # Set buffer size for network streams
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("❌ Failed to open stream")
            return False

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"✅ Connected successfully!")
        print(f"📐 Resolution: {width}x{height}")
        print(f"🎬 Stream FPS: {fps if fps > 0 else 'Unknown'}")
        print("-" * 60)

        return True

    def detect_vehicles(self, frame):
        """Detect vehicles using motion detection and contour analysis"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows
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

        # Filter and classify objects
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by minimum area (adjust for camera distance)
            if area < 300:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Simple classification based on size and aspect ratio
            if area > 5000:
                class_id = 1  # Large vehicle (truck/bus)
            elif area > 1500:
                class_id = 0  # Car
            elif aspect_ratio < 0.5 and area > 300:
                class_id = 3  # Pedestrian
            else:
                class_id = 2  # Bike/motorcycle

            detections.append([x, y, x+w, y+h, 0.8, class_id])

        if detections:
            detections_array = np.array(detections)
            return sv.Detections(
                xyxy=detections_array[:, :4],
                confidence=detections_array[:, 4],
                class_id=detections_array[:, 5].astype(int)
            )

        return sv.Detections.empty()

    def estimate_speed(self, track_id, positions):
        """Estimate vehicle speed from position history"""
        if len(positions) < 5:
            return 0

        # Calculate pixel displacement
        recent_positions = positions[-5:]
        distances = []
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            distances.append(np.sqrt(dx**2 + dy**2))

        avg_pixel_speed = np.mean(distances)

        # Convert to approximate km/h (needs calibration per camera)
        # This is a rough estimate - actual conversion depends on camera setup
        estimated_speed = avg_pixel_speed * 2  # Placeholder conversion

        return estimated_speed

    def process_frame(self, frame):
        """Process frame with vehicle detection and tracking"""
        # Detect vehicles
        detections = self.detect_vehicles(frame)

        # Track objects
        tracked = self.tracker.update_with_detections(detections)

        # Update tracking history and statistics
        if tracked.tracker_id is not None:
            for tracker_id, bbox, class_id in zip(
                tracked.tracker_id,
                tracked.xyxy,
                tracked.class_id if tracked.class_id is not None else [0] * len(tracked.tracker_id)
            ):
                # Calculate center point
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                self.track_history[tracker_id].append(center)

                # Limit history length
                if len(self.track_history[tracker_id]) > 30:
                    self.track_history[tracker_id].pop(0)

                # Update track info
                if tracker_id not in self.track_info:
                    self.track_info[tracker_id] = {
                        'first_seen': self.frame_count,
                        'last_seen': self.frame_count,
                        'class': class_id,
                        'positions': []
                    }

                    # Increment vehicle count
                    if class_id == 0:
                        self.vehicle_counts['cars'] += 1
                    elif class_id == 1:
                        self.vehicle_counts['trucks'] += 1
                    elif class_id == 2:
                        self.vehicle_counts['bikes'] += 1
                    elif class_id == 3:
                        self.vehicle_counts['pedestrians'] += 1
                    self.vehicle_counts['total'] += 1

                self.track_info[tracker_id]['last_seen'] = self.frame_count
                self.track_info[tracker_id]['positions'].append(center)

                # Estimate speed
                self.speed_estimates[tracker_id] = self.estimate_speed(
                    tracker_id,
                    self.track_history[tracker_id]
                )

        return tracked

    def annotate_frame(self, frame, detections):
        """Add annotations to frame"""
        annotated = frame.copy()

        # Draw bounding boxes
        if len(detections) > 0:
            annotated = self.box_annotator.annotate(
                scene=annotated,
                detections=detections
            )

            # Add custom labels
            if detections.tracker_id is not None:
                for i, (tracker_id, bbox, class_id) in enumerate(zip(
                    detections.tracker_id,
                    detections.xyxy,
                    detections.class_id if detections.class_id is not None else [0] * len(detections)
                )):
                    x1, y1, x2, y2 = bbox.astype(int)

                    # Determine vehicle type
                    vehicle_types = {0: "Car", 1: "Truck", 2: "Bike", 3: "Person"}
                    vehicle_type = vehicle_types.get(class_id, "Unknown")

                    # Get speed
                    speed = self.speed_estimates.get(tracker_id, 0)

                    # Create label
                    label = f"ID:{tracker_id} {vehicle_type}"
                    if speed > 5:  # Only show speed if moving
                        label += f" ~{speed:.0f}km/h"

                    # Draw label background
                    label_size = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        1
                    )[0]

                    cv2.rectangle(
                        annotated,
                        (x1, y1 - 20),
                        (x1 + label_size[0] + 5, y1),
                        (0, 0, 0),
                        -1
                    )

                    # Draw label text
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

        # Draw tracking trails
        for tracker_id, history in self.track_history.items():
            if len(history) > 1:
                points = np.array(history, dtype=np.int32)

                # Draw trail with gradient
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    color = (
                        int(255 * (1 - alpha)),  # Red decreases
                        int(255 * alpha),         # Green increases
                        128                       # Blue constant
                    )
                    cv2.line(
                        annotated,
                        tuple(points[i-1]),
                        tuple(points[i]),
                        color,
                        2
                    )

        # Add statistics overlay
        self.add_stats_overlay(annotated)

        return annotated

    def add_stats_overlay(self, frame):
        """Add statistics overlay to frame"""
        height, width = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Add text
        stats = [
            f"TRAFFIC MONITORING",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
            f"FPS: {self.fps:.1f}",
            f"",
            f"VEHICLE COUNT:",
            f"  Cars: {self.vehicle_counts['cars']}",
            f"  Trucks: {self.vehicle_counts['trucks']}",
            f"  Bikes: {self.vehicle_counts['bikes']}",
            f"  Pedestrians: {self.vehicle_counts['pedestrians']}",
            f"  Total: {self.vehicle_counts['total']}",
            f"",
            f"Active Tracks: {len([t for t in self.track_info.values() if self.frame_count - t['last_seen'] < 30])}"
        ]

        y = 30
        for stat in stats:
            if stat == "TRAFFIC MONITORING":
                cv2.putText(
                    frame, stat, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
            elif stat.startswith("  "):
                cv2.putText(
                    frame, stat, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                )
            elif stat == "":
                pass
            else:
                cv2.putText(
                    frame, stat, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
            y += 15

    def run(self, duration=None, save_video=False):
        """Main processing loop"""
        if not self.cap:
            print("❌ No stream connected")
            return

        self.start_time = time.time()

        # Setup video writer if saving
        writer = None
        if save_video:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 20

            output_file = f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            print(f"💾 Recording to: {output_file}")

        print("\n🚦 TRAFFIC MONITORING STARTED")
        print("Press Ctrl+C to stop\n")

        last_print_time = time.time()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Stream interrupted, reconnecting...")
                    time.sleep(1)
                    continue

                self.frame_count += 1

                # Calculate FPS
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0

                # Check duration
                if duration and elapsed > duration:
                    print(f"\n⏱️  Duration limit ({duration}s) reached")
                    break

                # Process frame
                detections = self.process_frame(frame)

                # Annotate frame
                annotated_frame = self.annotate_frame(frame, detections)

                # Save frame if recording
                if writer:
                    writer.write(annotated_frame)

                # Save snapshot every 30 seconds
                if self.frame_count % (30 * 20) == 0:  # Assuming ~20 fps
                    filename = f"traffic_snapshot_{self.frame_count:06d}.jpg"
                    cv2.imwrite(filename, annotated_frame)

                # Print stats every 5 seconds
                if time.time() - last_print_time > 5:
                    self.print_live_stats()
                    last_print_time = time.time()

                # Try to display (might not work headless)
                try:
                    cv2.imshow('Traffic Monitor', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    pass

        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping traffic monitor...")

        finally:
            self.cleanup(writer)

    def print_live_stats(self):
        """Print live statistics"""
        active_tracks = len([
            t for t in self.track_info.values()
            if self.frame_count - t['last_seen'] < 30
        ])

        print(f"\r🚦 Active: {active_tracks:3d} | "
              f"🚗 Cars: {self.vehicle_counts['cars']:4d} | "
              f"🚚 Trucks: {self.vehicle_counts['trucks']:3d} | "
              f"🏍️ Bikes: {self.vehicle_counts['bikes']:3d} | "
              f"🚶 Pedestrians: {self.vehicle_counts['pedestrians']:3d} | "
              f"FPS: {self.fps:.1f}", end="")

    def cleanup(self, writer=None):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Print final report
        print("\n\n" + "=" * 60)
        print("📊 TRAFFIC MONITORING REPORT")
        print("=" * 60)

        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Total Frames: {self.frame_count}")
        print(f"Average FPS: {self.fps:.1f}")

        print(f"\n🚦 TRAFFIC STATISTICS:")
        print(f"  Total Vehicles: {self.vehicle_counts['total']}")
        print(f"  - Cars: {self.vehicle_counts['cars']}")
        print(f"  - Trucks/Buses: {self.vehicle_counts['trucks']}")
        print(f"  - Motorcycles/Bikes: {self.vehicle_counts['bikes']}")
        print(f"  - Pedestrians: {self.vehicle_counts['pedestrians']}")

        # Calculate average speeds by vehicle type
        if self.track_info:
            speeds_by_type = defaultdict(list)
            for info in self.track_info.values():
                if len(info['positions']) > 5:
                    vehicle_class = info['class']
                    track_id = list(self.track_info.keys())[list(self.track_info.values()).index(info)]
                    if track_id in self.speed_estimates:
                        speeds_by_type[vehicle_class].append(self.speed_estimates[track_id])

            if speeds_by_type:
                print(f"\n📈 AVERAGE SPEEDS (approximate):")
                vehicle_types = {0: "Cars", 1: "Trucks", 2: "Bikes", 3: "Pedestrians"}
                for class_id, speeds in speeds_by_type.items():
                    if speeds:
                        avg_speed = np.mean([s for s in speeds if s > 5])
                        if not np.isnan(avg_speed):
                            print(f"  {vehicle_types.get(class_id, 'Unknown')}: ~{avg_speed:.0f} km/h")

def main():
    print("=" * 60)
    print("🚦 SUPER VISION - LIVE TRAFFIC MONITORING")
    print("=" * 60)

    # Public traffic camera streams
    traffic_streams = {
        '1': ('https://cctv.bremerton.wa.us/nph-h264.cgi/video.h264', 'Bremerton, WA - Warren Ave Bridge'),
        '2': ('http://webcam.mchcwa.org/airport2/GetData.cgi', 'Airport Traffic Camera'),
        '3': ('https://s52.nysdot.skyvdn.com/rtplive/R11_098/playlist.m3u8', 'NY State DOT Camera'),
        '4': ('http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard', 'Taiwan Traffic'),
        '5': ('custom', 'Enter custom URL')
    }

    print("\n📹 Available Traffic Cameras:")
    for key, (url, name) in traffic_streams.items():
        if url != 'custom':
            print(f"  {key}. {name}")
        else:
            print(f"  {key}. {name}")

    choice = input("\nSelect camera (1-5): ").strip()

    if choice == '5':
        stream_url = input("Enter stream URL: ").strip()
    elif choice in traffic_streams:
        stream_url, name = traffic_streams[choice]
        print(f"\n📍 Selected: {name}")
    else:
        # Default to a test stream
        stream_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        print("\n⚠️  Using test video stream")

    # Options
    save = input("\n💾 Save video output? (y/n): ").strip().lower() == 'y'
    duration_str = input("⏱️  Monitor duration in seconds (Enter for unlimited): ").strip()
    duration = int(duration_str) if duration_str else None

    # Create tracker
    tracker = TrafficTracker()

    # Connect and run
    if tracker.connect_to_stream(stream_url):
        tracker.run(duration=duration, save_video=save)
    else:
        print("❌ Failed to connect to traffic stream")
        print("\n💡 Tips:")
        print("  - Try a different camera from the list")
        print("  - Check if the stream URL is accessible")
        print("  - Some cameras may require VPN or specific regions")
        sys.exit(1)

if __name__ == "__main__":
    main()