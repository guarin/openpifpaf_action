import numpy as np


def bbox_area(bbox):
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    assert (w >= 0) and (h >= 0)
    return w * h


def bbox_center(bbox):
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return [x + w / 2, y + h / 2]


def keypoint_center(keypoints, keypoint_indices):
    keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 3)
    keypoints = keypoints[keypoint_indices, :2]
    return keypoints.mean(0).tolist()


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    minx = max(x1, x2)
    maxx = min(x1 + w1, x2 + w2)
    miny = max(y1, y2)
    maxy = min(y1 + h1, y2 + h2)
    w = maxx - minx
    h = maxy - miny
    if (w < 0) or (h < 0):
        intersection = 0
    else:
        intersection = w * h
    a1 = w1 * h1
    a2 = w2 * h2
    iou = intersection / (a1 + a2 - intersection)
    return iou


def index_dict(list):
    return {value: index for index, value in enumerate(list)}
