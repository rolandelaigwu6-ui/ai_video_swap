"""Authorized synthetic media research tools."""
import os

# Must be set before OpenCV is imported. Limits a decoded image to ~48 MB RGB.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "16000000")
