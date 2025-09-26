#!/usr/bin/env python3
"""
Sports Tracking Example
Optimized settings for tracking players and balls in various sports
"""

import sys
sys.path.append('../..')
from src.core.motion_tracker import MotionTracker
import cv2


def track_soccer_match():
    """Track soccer/football match with optimized settings"""

    # Player tracking settings
    player_tracker = MotionTracker(
        detection_threshold=500,   # Medium-large objects (players)
        trail_length=30            # Movement analysis trail
    )

    # Configure background subtractor for field conditions
    player_tracker.bg_subtractor.setHistory(300)
    player_tracker.bg_subtractor.setVarThreshold(20)

    # Connect to video source
    player_tracker.connect_source("soccer_match.mp4")

    # Run tracking
    stats = player_tracker.run(duration=5400, save_interval=300)  # 90 minutes

    print(f"Total players tracked: {stats['total_objects']}")
    print(f"Processing FPS: {stats['fps']:.1f}")

    return stats


def track_basketball_game():
    """Track basketball game with court-optimized settings"""

    tracker = MotionTracker(
        detection_threshold=600,   # Players on indoor court
        trail_length=25            # Quick movements
    )

    # Adjust for indoor lighting
    tracker.bg_subtractor.setHistory(200)
    tracker.bg_subtractor.setVarThreshold(16)

    tracker.connect_source("basketball_game.mp4")

    # Process game
    stats = tracker.run(duration=2880, save_interval=120)  # 48 minutes

    return stats


def track_tennis_match():
    """Track tennis match with settings for players and ball"""

    # Player tracker
    player_tracker = MotionTracker(
        detection_threshold=400,   # Two players
        trail_length=20            # Short trails for quick movements
    )

    # Ball tracker (separate instance for small object)
    ball_tracker = MotionTracker(
        detection_threshold=75,    # Very small object
        trail_length=100          # Long trail for trajectory analysis
    )

    # Fine-tune ball detection
    ball_tracker.bg_subtractor.setHistory(100)
    ball_tracker.bg_subtractor.setVarThreshold(10)

    # Connect both to same source
    player_tracker.connect_source("tennis_match.mp4")
    ball_tracker.connect_source("tennis_match.mp4")

    # Run both trackers
    player_stats = player_tracker.run(duration=7200)  # 2 hours
    ball_stats = ball_tracker.run(duration=7200)

    return player_stats, ball_stats


def track_swimming_race():
    """Track swimmers in pool with water-specific settings"""

    tracker = MotionTracker(
        detection_threshold=300,   # Partially visible swimmers
        trail_length=40            # Lane tracking
    )

    # Adjust for water reflections and splashing
    tracker.bg_subtractor.setHistory(150)
    tracker.bg_subtractor.setVarThreshold(30)  # Higher threshold for water
    tracker.bg_subtractor.setDetectShadows(False)  # Disable shadow detection

    tracker.connect_source("swimming_race.mp4")

    stats = tracker.run(save_interval=60)

    # Analyze lane performance
    lane_data = tracker.get_object_data()
    for swimmer_id, data in lane_data.items():
        positions = data.get('positions', [])
        if positions:
            # Calculate average lane position
            avg_x = sum(p[0] for p in positions) / len(positions)
            print(f"Swimmer {swimmer_id}: Average lane position X={avg_x:.1f}")

    return stats


def track_track_and_field():
    """Track runners on athletics track"""

    tracker = MotionTracker(
        detection_threshold=450,   # Multiple runners
        trail_length=50            # Full lap tracking
    )

    # Outdoor settings
    tracker.bg_subtractor.setHistory(400)
    tracker.bg_subtractor.setVarThreshold(18)

    tracker.connect_source("track_race.mp4")

    stats = tracker.run(save_interval=30)

    # Analyze lap times
    runner_data = tracker.get_object_data()
    for runner_id, data in runner_data.items():
        first_seen = data.get('first_seen', 0)
        last_seen = data.get('last_seen', 0)
        duration = (last_seen - first_seen) / 30  # Assuming 30 FPS
        print(f"Runner {runner_id}: Visible for {duration:.2f} seconds")

    return stats


def main():
    """Example usage of sports tracking"""

    print("Sports Tracking System")
    print("-" * 40)
    print("Select sport to track:")
    print("1. Soccer/Football")
    print("2. Basketball")
    print("3. Tennis")
    print("4. Swimming")
    print("5. Track & Field")

    choice = input("\nEnter choice (1-5): ")

    if choice == "1":
        track_soccer_match()
    elif choice == "2":
        track_basketball_game()
    elif choice == "3":
        track_tennis_match()
    elif choice == "4":
        track_swimming_race()
    elif choice == "5":
        track_track_and_field()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()