#!/usr/bin/env python3
"""
Super Vision - Jackson Hole Town Square Live Traffic Analysis
Real traffic monitoring from Wyoming, USA
"""

import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
import time
from datetime import datetime

print("🚦 SUPER VISION - JACKSON HOLE LIVE TRAFFIC")
print("=" * 60)
print("📍 Location: Jackson Hole Town Square, Wyoming, USA")
print("🎥 Source: Live YouTube Stream")
print("=" * 60)

# Direct stream URL from Jackson Hole Town Square
STREAM_URL = "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1758852465/ei/EaHVaLanHtnAmLAPlojt2AQ/ip/78.142.65.188/id/1EiC9bvVGnk.7/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/rqh/1/hls_chunk_host/rr1---sn-puigxxc-8pxl.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/gcr/at/bui/ATw7iSUmgLelNokmCMOmVqhXdl8lgqN1yvAr2vjOcKji-cKW8hq0b3uUmp38jhaOKUft5i9VjDrLmcYQ/spc/hcYD5WTp4qlxZZRZ12rg/vprv/1/playlist_type/DVR/initcwndbps/3372500/met/1758830866,/mh/hd/mm/44/mn/sn-puigxxc-8pxl/ms/lva/mv/m/mvi/1/pl/23/rms/lva,lva/dover/11/pacing/0/keepalive/yes/fexp/51552689,51565115,51565681,51580968/mt/1758830458/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,xpc,gcr,bui,spc,vprv,playlist_type/sig/AJfQdSswRQIgSxCf4xqdtlX5NHT_KHZNmGl6ivEtI5PfgEBS6JRlEdMCIQCFH3CECRxbgZyp6mytjKLa72LErhvX5rg-kw9fItJSVw%3D%3D/lsparams/hls_chunk_host,initcwndbps,met,mh,mm,mn,ms,mv,mvi,pl,rms/lsig/APaTxxMwRQIhAKqfso1SnfwZ3uNovSQ25qfI_lR8lYRyy_nJQjn8Vgy-AiBZ25TAO-TkJ_q5kSOePNeYv9UVjhhJhm7u_cCC1CBD_A%3D%3D/playlist/index.m3u8"

print("\n🔗 Connecting to Jackson Hole live stream...")

# Initialize components
cap = cv2.VideoCapture(STREAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Check connection
if not cap.isOpened():
    print("❌ Failed to connect to stream")
    print("⚠️  Stream URL may have expired. Getting fresh URL...")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✅ Connected! Resolution: {width}x{height}")

# Initialize tracking
box_annotator = sv.BoxAnnotator(thickness=2)
tracker = sv.ByteTrack()
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    detectShadows=True,
    varThreshold=25,
    history=200
)

# Tracking data
track_history = defaultdict(lambda: deque(maxlen=30))
vehicle_stats = {
    'cars': 0,
    'trucks': 0,
    'pedestrians': 0,
    'total': 0
}

frame_count = 0
start_time = time.time()

print("\n🚦 MONITORING JACKSON HOLE TRAFFIC")
print("Analyzing for 30 seconds...\n")

try:
    while time.time() - start_time < 30:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours (vehicles/people)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and create detections
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio
                if 0.3 < w/h < 4:
                    boxes.append([x, y, x+w, y+h])

        # Create detections
        if boxes:
            boxes = np.array(boxes)
            detections = sv.Detections(
                xyxy=boxes,
                confidence=np.ones(len(boxes)) * 0.9,
                class_id=np.zeros(len(boxes), dtype=int)
            )
        else:
            detections = sv.Detections.empty()

        # Track objects
        tracked = tracker.update_with_detections(detections)

        # Update tracking history
        if tracked.tracker_id is not None:
            for tracker_id, bbox in zip(tracked.tracker_id, tracked.xyxy):
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                track_history[tracker_id].append(center)

                # Count new vehicles
                if tracker_id > vehicle_stats['total']:
                    vehicle_stats['total'] = tracker_id

                    # Classify by size
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > 10000:
                        vehicle_stats['trucks'] += 1
                    elif area > 3000:
                        vehicle_stats['cars'] += 1
                    else:
                        vehicle_stats['pedestrians'] += 1

        # Annotate frame
        annotated = frame.copy()
        if len(tracked) > 0:
            annotated = box_annotator.annotate(annotated, tracked)

            # Draw tracking trails
            for tracker_id, history in track_history.items():
                if len(history) > 2:
                    points = np.array(history, dtype=np.int32)
                    for i in range(1, len(points)):
                        alpha = i / len(points)
                        color = (int(255 * (1-alpha)), int(255 * alpha), 0)
                        cv2.line(annotated, tuple(points[i-1]), tuple(points[i]), color, 2)

        # Add overlay
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Dashboard background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # Add text
        cv2.putText(annotated, "JACKSON HOLE LIVE", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated, f"Time: {datetime.now().strftime('%H:%M:%S')} MST", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Active Objects: {len(tracked)}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, f"Total Tracked: {vehicle_stats['total']}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, f"Cars: {vehicle_stats['cars']} | Trucks: {vehicle_stats['trucks']}", (20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Pedestrians: {vehicle_stats['pedestrians']}", (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS
        cv2.putText(annotated, f"FPS: {fps:.1f}", (width-120, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save snapshots
        if frame_count % 150 == 0:  # Every ~5 seconds at 30fps
            filename = f"jackson_hole_{frame_count:06d}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"💾 Saved: {filename} | Active: {len(tracked)} | Total: {vehicle_stats['total']}")

        # Display stats every 3 seconds
        if frame_count % 90 == 0:
            print(f"📊 Live: Active: {len(tracked):2d} | Cars: {vehicle_stats['cars']:3d} | "
                  f"Trucks: {vehicle_stats['trucks']:2d} | Pedestrians: {vehicle_stats['pedestrians']:3d} | "
                  f"FPS: {fps:.1f}")

        # Try to display
        try:
            cv2.imshow('Jackson Hole Traffic Monitor', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass

except KeyboardInterrupt:
    print("\n⏹️ Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Final report
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("📊 JACKSON HOLE TRAFFIC ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Duration: {elapsed:.1f} seconds")
    print(f"Frames Processed: {frame_count}")
    print(f"Average FPS: {frame_count/elapsed:.1f}")
    print(f"\n🚦 TRAFFIC STATISTICS:")
    print(f"  Total Objects Tracked: {vehicle_stats['total']}")
    print(f"  Cars: {vehicle_stats['cars']}")
    print(f"  Trucks/Large Vehicles: {vehicle_stats['trucks']}")
    print(f"  Pedestrians: {vehicle_stats['pedestrians']}")
    print("\n✅ Analysis complete!")