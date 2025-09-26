#!/usr/bin/env python3

# Public video streams for testing
PUBLIC_STREAMS = {
    'webcams': [
        {
            'name': 'Times Square Live',
            'url': 'https://www.youtube.com/watch?v=F109TZt3nRc',
            'type': 'youtube',
            'description': 'Live webcam from Times Square, NYC'
        },
        {
            'name': 'Abbey Road Crossing',
            'url': 'https://www.youtube.com/watch?v=KGuCGd726RA',
            'type': 'youtube',
            'description': 'Famous Beatles crossing in London'
        }
    ],
    'test_videos': [
        {
            'name': 'Big Buck Bunny',
            'url': 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
            'type': 'mp4',
            'description': 'Open source animated film'
        },
        {
            'name': 'Mux Test Stream',
            'url': 'https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8',
            'type': 'hls',
            'description': 'HLS test stream from Mux'
        },
        {
            'name': 'Sample Traffic Video',
            'url': 'https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4',
            'type': 'mp4',
            'description': 'Short sample video'
        }
    ],
    'rtsp_examples': [
        {
            'name': 'RTSP Test Pattern',
            'url': 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4',
            'type': 'rtsp',
            'description': 'RTSP test stream'
        }
    ]
}

# Processing settings
PROCESSING_CONFIG = {
    'edge_detection': {
        'low_threshold': 50,
        'high_threshold': 150
    },
    'motion_detection': {
        'threshold': 25,
        'min_area': 500
    },
    'display': {
        'window_width': 1280,
        'window_height': 720,
        'font_scale': 0.7,
        'font_color': (0, 255, 0),
        'font_thickness': 2
    }
}

# Performance settings
PERFORMANCE_CONFIG = {
    'frame_queue_size': 10,
    'max_fps': 30,
    'buffer_size': 1024,
    'reconnect_delay': 5,
    'max_reconnect_attempts': 3
}