# 🎯 Motion Tracking with Supervision

A powerful, universal motion tracking system using computer vision for real-time object detection and tracking. Perfect for traffic monitoring, sports analysis, robotics competitions, and any motion analysis project.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![Supervision](https://img.shields.io/badge/Supervision-0.20%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

- **Real-time Object Detection & Tracking** - Track multiple objects simultaneously
- **Motion Trail Visualization** - See movement patterns over time
- **Universal Application** - Works with any video source (webcam, files, streams)
- **Live Statistics** - FPS, object count, tracking IDs
- **Easy to Use** - Simple API for custom applications

## 📋 Applications

- 🚦 **Traffic Monitoring** - Count vehicles, analyze traffic flow
- 🏃 **Sports Analysis** - Track players, analyze movements
- 🤖 **Robotics/FLL** - Track robot movements, analyze patterns
- 🏎️ **Esports/Gaming** - Track game elements (e.g., Trackmania cars)
- 📹 **Security** - Monitor areas for motion detection
- 🔬 **Research** - Analyze any moving objects

## 🛠️ Installation (macOS)

### Prerequisites

1. **Python 3.8+** installed
2. **Homebrew** (for system dependencies)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/gbechtold/motiontracking.git
cd motiontracking

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### For YouTube Live Streams (Optional)

```bash
# Install yt-dlp for YouTube stream support
pip install yt-dlp
```

## 🎮 Quick Start

### Basic Motion Tracking

```python
from src.core.motion_tracker import MotionTracker

# Initialize tracker
tracker = MotionTracker()

# Connect to webcam
tracker.connect_source(0)  # 0 for default webcam

# Run tracking for 30 seconds
tracker.run(duration=30, save_interval=150)
```

### Track a Video File

```python
# Track any video file
tracker.connect_source("path/to/your/video.mp4")
tracker.run()
```

### Track Live Stream

```python
# Track YouTube live stream
import subprocess

# Get stream URL
result = subprocess.run(['yt-dlp', '-g', 'YOUTUBE_URL'], capture_output=True, text=True)
stream_url = result.stdout.strip()

tracker.connect_source(stream_url)
tracker.run(duration=60)
```

## 📁 Project Structure

```
motiontracking/
├── src/
│   ├── core/              # Core tracking modules
│   │   ├── motion_tracker.py
│   │   └── base_tracker.py
│   ├── trackers/          # Specialized trackers
│   └── utils/             # Utility functions
├── examples/
│   ├── traffic/           # Traffic monitoring examples
│   ├── sports/            # Sports tracking examples
│   └── robotics/          # FLL/robotics examples
├── output/                # Saved frames and videos
├── docs/                  # Documentation
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md
```

## 💡 Examples

### 1. Traffic Monitoring

```bash
# Run traffic analysis on a street cam
python examples/traffic/traffic_tracker.py
```

### 2. Sports Tracking

```python
from src.core.motion_tracker import MotionTracker

tracker = MotionTracker(
    detection_threshold=300,  # Smaller objects
    trail_length=50           # Longer trails
)
tracker.connect_source("sports_video.mp4")
tracker.run()
```

### 3. Robot Tracking (FLL)

```python
# Track LEGO robots on FLL table
tracker = MotionTracker(detection_threshold=200)
tracker.connect_source(0)  # Overhead camera
stats = tracker.run(duration=150)  # 2.5 minute match
print(f"Robot movements: {stats['total_objects']}")
```

## ⚙️ Configuration

### Tracker Parameters

- `detection_threshold`: Minimum object size (pixels)
- `trail_length`: Number of positions to keep in trail
- `tracker_type`: "bytetrack" (recommended)

### Detection Tuning

```python
# For small objects (e.g., balls, LEGO robots)
tracker = MotionTracker(detection_threshold=200)

# For large objects (e.g., vehicles)
tracker = MotionTracker(detection_threshold=1000)

# For fast moving objects
tracker = MotionTracker(trail_length=50)
```

## 📊 Output

The system provides:
- **Live visualization** with bounding boxes and trails
- **Real-time statistics** (FPS, object count)
- **Saved frames** at specified intervals
- **Tracking data** accessible via API

## 🎯 Use Cases

### Traffic Analysis
- Vehicle counting
- Speed estimation
- Traffic flow patterns
- Intersection monitoring

### Sports Analytics
- Player tracking
- Ball trajectory
- Movement heatmaps
- Performance metrics

### Robotics Competitions
- Robot path tracking
- Strategy analysis
- Performance optimization
- Collision detection

### Esports/Gaming
- Player movement patterns
- Racing line analysis
- Strategy visualization

## 🐛 Troubleshooting

### Common Issues

1. **"No module named cv2"**
   ```bash
   pip install opencv-python
   ```

2. **Camera not working**
   - Check camera permissions in System Preferences
   - Try different camera index (0, 1, 2)

3. **Stream connection failed**
   - Check internet connection
   - Update stream URL
   - Install yt-dlp for YouTube streams

4. **Low FPS**
   - Reduce resolution
   - Increase detection threshold
   - Use fewer visual effects

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Supervision](https://github.com/roboflow/supervision) - Amazing tracking library
- [OpenCV](https://opencv.org/) - Computer vision foundation
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - SOTA tracking algorithm

## 📧 Contact

Guntram Bechtold - [GitHub](https://github.com/gbechtold)

Project Link: [https://github.com/gbechtold/motiontracking](https://github.com/gbechtold/motiontracking)

---

**Made with ❤️ for the motion tracking community**