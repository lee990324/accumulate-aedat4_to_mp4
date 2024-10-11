import numpy as np
import cv2
import dv

# aedat data path
aedat_file_path = 'C:/Users/gram15/Desktop/20241010/test2.aedat4'

with dv.AedatFile(aedat_file_path) as f:
    events = np.hstack([packet for packet in f['events'].numpy()])

# Function to convert events to frames
def events_to_frames(events, width, height, time_window, accumulate=33):
    frames = []
    buffer = [np.zeros((height, width), dtype=np.uint8) for _ in range(accumulate)]
    frame_start_time = events['timestamp'][0]

    for event in events:
        if event['timestamp'] - frame_start_time > time_window:
            frames.append(buffer[0].copy())
            buffer = buffer[1:] + [np.zeros((height, width), dtype=np.uint8)]
            frame_start_time = event['timestamp']

        x, y = event['x'], event['y']
        polarity = event['polarity']

        # Accumulate pixel value based on polarity
        if polarity == 1:
            for frame in buffer:
                frame[y, x] = min(frame[y, x] + 50, 255)

    frames.append(buffer[0])  # Append the last frame
    return frames

# Set frame size and time window
width, height = 346, 260  # Example values
time_window = 10000  # 10ms
#10ms - 10fps / 3.3ms - 30fps

# Convert events to frames
frames = events_to_frames(events, width, height, time_window)

# Flip each frame vertically
flipped_frames = [cv2.flip(frame, -1) for frame in frames]

# Video saving settings
video_file = 'C:/Users/gram15/Desktop/20241010/test2_acc.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
fps = 100  # Frame rate
frame_size = (width, height)

# Create VideoWriter object
out = cv2.VideoWriter(video_file, fourcc, fps, frame_size, isColor=False)

# Write frames to video
for frame in flipped_frames:
    out.write(frame)

# Release the VideoWriter object
out.release()