"""
Topic definitions for the Phase 3A camera-only baseline.
These are locked to the current RealSense D435i setup.
"""

COLOR_IMAGE_TOPIC = "/camera/camera/color/image_raw"
ALIGNED_DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
COLOR_CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
DETECTIONS_TOPIC = "/camera_person_tracker/detections"

# Synced versions (recommended for accurate depth)
SYNCED_DETECTIONS_TOPIC = "/camera_person_tracker/synced_detections"
SYNCED_ANNOTATED_IMAGE_TOPIC = "/camera_person_tracker/synced_annotated_image"
SYNCED_TARGET_IMAGE_TOPIC = "/camera_person_tracker/synced_target_image"
