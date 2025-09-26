#!/usr/bin/env python3
"""
Robotics Competition Tracking
Optimized for tracking robots in competition environments
"""

import sys
sys.path.append('../..')
from src.core.motion_tracker import MotionTracker
import cv2
import numpy as np
from collections import defaultdict


class RoboticsTracker(MotionTracker):
    """Extended tracker for robotics competitions"""

    def __init__(self, arena_size="standard", robot_count=2):
        """
        Initialize robotics tracker

        Args:
            arena_size: "small" (1x1m), "standard" (2.4x2.4m), "large" (3x3m+)
            robot_count: Expected number of robots
        """
        # Set detection threshold based on arena size
        thresholds = {
            "small": 200,
            "standard": 350,
            "large": 500
        }

        super().__init__(
            detection_threshold=thresholds.get(arena_size, 350),
            trail_length=60  # Longer trails for strategy analysis
        )

        self.arena_size = arena_size
        self.robot_count = robot_count
        self.robot_strategies = defaultdict(list)

    def analyze_strategy(self, robot_id, positions):
        """Analyze robot movement strategy"""
        if len(positions) < 10:
            return "Initializing"

        # Calculate movement variance
        x_positions = [p[0] for p in positions]
        y_positions = [p[1] for p in positions]

        x_variance = np.var(x_positions)
        y_variance = np.var(y_positions)

        # Classify strategy based on movement patterns
        if x_variance < 100 and y_variance < 100:
            return "Stationary/Defensive"
        elif x_variance > 1000 or y_variance > 1000:
            return "Exploratory/Aggressive"
        else:
            return "Balanced/Strategic"

    def detect_interactions(self, tracked_objects):
        """Detect robot interactions (collisions, near-misses)"""
        interactions = []

        if tracked_objects.tracker_id is None:
            return interactions

        positions = tracked_objects.xyxy

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                # Calculate distance between robots
                dist = np.linalg.norm(
                    positions[i][:2] - positions[j][:2]
                )

                if dist < 100:  # Proximity threshold in pixels
                    interactions.append({
                        'robot1': tracked_objects.tracker_id[i],
                        'robot2': tracked_objects.tracker_id[j],
                        'distance': dist,
                        'type': 'collision' if dist < 50 else 'near-miss'
                    })

        return interactions


def track_vex_competition():
    """Track VEX Robotics Competition"""

    tracker = RoboticsTracker(arena_size="standard", robot_count=4)

    # Configure for VEX field conditions
    tracker.bg_subtractor.setHistory(200)
    tracker.bg_subtractor.setVarThreshold(15)

    # Connect to overhead camera
    tracker.connect_source(0)  # Or video file

    print("Tracking VEX Competition")
    print("-" * 40)

    frame_count = 0
    interaction_log = []

    # Run for match duration (2 minutes autonomous + driver)
    start_time = cv2.getTickCount()
    duration = 120  # seconds

    while True:
        ret, frame = tracker.cap.read()
        if not ret:
            break

        frame_count += 1

        # Process frame
        detections = tracker.detect_motion(frame)
        tracked = tracker.update_tracking(detections)

        # Analyze strategies
        if frame_count % 30 == 0:  # Every second (assuming 30 FPS)
            for robot_id in tracker.robot_data.keys():
                positions = tracker.track_history[robot_id]
                strategy = tracker.analyze_strategy(robot_id, positions)
                tracker.robot_strategies[robot_id].append(strategy)

        # Check for interactions
        interactions = tracker.detect_interactions(tracked)
        if interactions:
            interaction_log.extend(interactions)

        # Annotate frame
        annotated = tracker.annotate_frame(frame, tracked)

        # Add competition overlay
        cv2.putText(annotated, "VEX Robotics Tracking", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        remaining = max(0, duration - elapsed)
        cv2.putText(annotated, f"Time: {remaining:.1f}s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Display
        cv2.imshow('VEX Competition Tracker', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q') or elapsed > duration:
            break

    tracker.cleanup()

    # Print analysis
    print("\nCompetition Analysis:")
    for robot_id, strategies in tracker.robot_strategies.items():
        if strategies:
            most_common = max(set(strategies), key=strategies.count)
            print(f"Robot {robot_id}: Primary strategy - {most_common}")

    print(f"\nTotal interactions detected: {len(interaction_log)}")


def track_frc_competition():
    """Track FIRST Robotics Competition (FRC)"""

    tracker = RoboticsTracker(arena_size="large", robot_count=6)

    # FRC specific settings (larger robots, faster movement)
    tracker.detection_threshold = 800
    tracker.bg_subtractor.setHistory(300)

    tracker.connect_source("frc_match.mp4")

    print("Tracking FRC Match")
    print("-" * 40)

    stats = tracker.run(duration=150, save_interval=150)  # 2.5 minute match

    # Analyze robot paths for scoring zones
    robot_data = tracker.get_object_data()

    for robot_id, data in robot_data.items():
        positions = data.get('positions', [])
        if positions:
            # Check time spent in different zones
            # This would require field calibration
            print(f"Robot {robot_id}: {len(positions)} tracking points")


def track_sumo_robots():
    """Track robot sumo competition"""

    tracker = RoboticsTracker(arena_size="small", robot_count=2)

    # Sumo specific settings (small arena, quick movements)
    tracker.detection_threshold = 150
    tracker.trail_length = 30

    # High sensitivity for quick movements
    tracker.bg_subtractor.setVarThreshold(10)
    tracker.bg_subtractor.setHistory(100)

    tracker.connect_source("sumo_match.mp4")

    print("Tracking Sumo Robot Match")
    print("-" * 40)

    # Track for typical sumo match (3 minutes)
    stats = tracker.run(duration=180)

    # Analyze aggression patterns
    robot_data = tracker.get_object_data()

    for robot_id, data in robot_data.items():
        positions = data.get('positions', [])
        if len(positions) > 10:
            # Calculate average speed
            speeds = []
            for i in range(1, len(positions)):
                dist = np.linalg.norm(
                    np.array(positions[i]) - np.array(positions[i-1])
                )
                speeds.append(dist)

            avg_speed = np.mean(speeds)
            print(f"Robot {robot_id}: Average speed = {avg_speed:.2f} pixels/frame")


def track_line_following():
    """Track line-following robots"""

    tracker = MotionTracker(
        detection_threshold=100,   # Small robots
        trail_length=100          # Long trail to see full path
    )

    # Configure for track conditions
    tracker.bg_subtractor.setHistory(500)
    tracker.bg_subtractor.setVarThreshold(20)

    tracker.connect_source("line_following.mp4")

    print("Tracking Line Following Robots")
    print("-" * 40)

    stats = tracker.run(save_interval=60)

    # Analyze path accuracy
    robot_data = tracker.get_object_data()

    for robot_id, data in robot_data.items():
        positions = data.get('positions', [])
        if len(positions) > 20:
            # Calculate path smoothness (lower variance = smoother)
            x_positions = [p[0] for p in positions]
            y_positions = [p[1] for p in positions]

            x_smoothness = np.std(np.diff(x_positions))
            y_smoothness = np.std(np.diff(y_positions))

            print(f"Robot {robot_id}: Path smoothness X={x_smoothness:.2f}, Y={y_smoothness:.2f}")


def main():
    """Example usage for robotics tracking"""

    print("Robotics Competition Tracker")
    print("-" * 40)
    print("Select competition type:")
    print("1. VEX Robotics")
    print("2. FIRST Robotics (FRC)")
    print("3. Robot Sumo")
    print("4. Line Following")
    print("5. Custom Arena")

    choice = input("\nEnter choice (1-5): ")

    if choice == "1":
        track_vex_competition()
    elif choice == "2":
        track_frc_competition()
    elif choice == "3":
        track_sumo_robots()
    elif choice == "4":
        track_line_following()
    elif choice == "5":
        # Custom settings
        arena = input("Arena size (small/standard/large): ")
        robots = int(input("Number of robots: "))

        tracker = RoboticsTracker(arena_size=arena, robot_count=robots)
        source = input("Video source (0 for webcam, or file path): ")

        if source == "0":
            tracker.connect_source(0)
        else:
            tracker.connect_source(source)

        tracker.run(duration=120)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()