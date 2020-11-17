import numpy as np


def bbox_center(bbox):
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return [x + w / 2, y + h / 2]


def keypoint_center(keypoints, keypoint_indices):
    keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 3)
    keypoints = keypoints[keypoint_indices, :2]
    return keypoints.mean(0).tolist()
