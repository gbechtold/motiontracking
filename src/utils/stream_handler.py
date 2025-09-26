#!/usr/bin/env python3
import cv2
import time
import threading
from urllib.parse import urlparse
import subprocess
import sys

class StreamHandler:
    """Advanced stream handler with support for various protocols"""

    def __init__(self, url, reconnect_attempts=3):
        self.url = url
        self.reconnect_attempts = reconnect_attempts
        self.cap = None
        self.is_connected = False
        self.stream_type = self._detect_stream_type(url)

    def _detect_stream_type(self, url):
        """Detect stream type from URL"""
        parsed = urlparse(url)

        if url.endswith('.m3u8'):
            return 'hls'
        elif url.endswith('.mp4'):
            return 'mp4'
        elif parsed.scheme == 'rtsp':
            return 'rtsp'
        elif 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
            return 'youtube'
        else:
            return 'unknown'

    def connect(self):
        """Connect to stream with automatic retry"""
        for attempt in range(self.reconnect_attempts):
            try:
                print(f"🔄 Connection attempt {attempt + 1}/{self.reconnect_attempts}")

                if self.stream_type == 'youtube':
                    # For YouTube, we'd need youtube-dl/yt-dlp
                    print("⚠️  YouTube streams require yt-dlp. Using fallback URL.")
                    self.cap = cv2.VideoCapture(self.url)
                else:
                    # Direct connection for other streams
                    self.cap = cv2.VideoCapture(self.url)

                # Set buffer size for network streams
                if self.stream_type in ['hls', 'rtsp']:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if self.cap.isOpened():
                    self.is_connected = True
                    print("✅ Stream connected successfully!")
                    return True

            except Exception as e:
                print(f"❌ Connection failed: {e}")

            if attempt < self.reconnect_attempts - 1:
                time.sleep(2)

        return False

    def read_frame(self):
        """Read a single frame from the stream"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def get_stream_info(self):
        """Get stream information"""
        if not self.cap or not self.cap.isOpened():
            return None

        info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'codec': self.cap.get(cv2.CAP_PROP_FOURCC),
            'stream_type': self.stream_type
        }
        return info

    def release(self):
        """Release stream resources"""
        if self.cap:
            self.cap.release()
            self.is_connected = False


class StreamManager:
    """Manage multiple streams"""

    def __init__(self):
        self.streams = {}
        self.active_stream = None

    def add_stream(self, name, url):
        """Add a new stream"""
        handler = StreamHandler(url)
        if handler.connect():
            self.streams[name] = handler
            if not self.active_stream:
                self.active_stream = name
            return True
        return False

    def switch_stream(self, name):
        """Switch to a different stream"""
        if name in self.streams:
            self.active_stream = name
            return True
        return False

    def get_current_frame(self):
        """Get frame from active stream"""
        if self.active_stream and self.active_stream in self.streams:
            return self.streams[self.active_stream].read_frame()
        return None

    def cleanup(self):
        """Clean up all streams"""
        for stream in self.streams.values():
            stream.release()
        self.streams.clear()
        self.active_stream = None