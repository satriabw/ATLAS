"""Minimal test of video loading."""
import cv2
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

video_path = Path('/home/satriabw/Track2Data/data/raw/video/video_001.avi')

logger.info(f"Testing video: {video_path}")
logger.info(f"File exists: {video_path.exists()}")

# Test 1: Can we open it?
logger.info("\n=== Test 1: Open Video ===")
cap = cv2.VideoCapture(str(video_path))
if cap.isOpened():
    logger.info("✓ Video opened successfully")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  FPS: {fps}")
else:
    logger.error("✗ Failed to open video")
    exit(1)
cap.release()

# Test 2: Read some frames
logger.info("\n=== Test 2: Read Frames ===")
cap = cv2.VideoCapture(str(video_path))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frames_read = 0
for i in range(10):
    ret, frame = cap.read()
    if ret:
        frames_read += 1
    else:
        logger.warning(f"Failed to read frame {i}")
        break

logger.info(f"✓ Read {frames_read}/10 frames")
cap.release()

# Test 3: Multiple open/close cycles (simulating DataLoader)
logger.info("\n=== Test 3: Multiple Open/Close Cycles ===")
for cycle in range(5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"✗ Cycle {cycle}: Failed to open")
        break
    
    # Read a few frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"✓ Cycle {cycle} completed")

logger.info("\n=== All Video Tests Passed ===")
