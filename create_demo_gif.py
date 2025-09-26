#!/usr/bin/env python3
"""
Create demo GIF from Jackson Hole tracking frames
"""

import imageio
import os
from PIL import Image

# Get all jackson frames
frames = [
    "output/jackson_00150.jpg",
    "output/jackson_00300.jpg",
    "output/jackson_00450.jpg",
    "output/jackson_00600.jpg"
]

# Check which files exist
existing_frames = [f for f in frames if os.path.exists(f)]

if not existing_frames:
    print("No Jackson frames found in output directory")
    exit(1)

print(f"Found {len(existing_frames)} frames")

# Load and resize images
images = []
target_width = 800  # Resize for smaller GIF size

for frame_path in existing_frames:
    print(f"Processing: {frame_path}")
    img = Image.open(frame_path)

    # Calculate new height to maintain aspect ratio
    width_percent = target_width / float(img.size[0])
    target_height = int(float(img.size[1]) * width_percent)

    # Resize image
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    images.append(img)

# Add frames in reverse to create loop effect
all_images = images + images[::-1]

# Save as GIF
output_path = "docs/demo_tracking.gif"
os.makedirs("docs", exist_ok=True)

print(f"Creating GIF with {len(all_images)} frames...")

# Convert PIL images to numpy arrays for imageio
import numpy as np
frames_array = [np.array(img) for img in all_images]

# Create GIF with imageio
imageio.mimsave(output_path, frames_array, duration=0.5, loop=0)

print(f"GIF saved to: {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")