import json
import numpy as np
import matplotlib

from openpifpaf_action import utils


def load_json(file):
    """Loads an annotation file and filters for action annotations"""
    anns = json.load(open(file))
    filename = file.split("/")[-1][:-17]
    anns = [
        a
        for a in anns
        if ("action_probabilities" in a)
        and (a["keypoint_data"]["score"] > 0.01)
        and (any(a["action_probabilities"]))
    ]
    anns = list(sorted(anns, key=lambda a: a["keypoint_data"]["score"], reverse=True))
    return filename, anns


def score_fun(ann, kp_ann):
    """Scoring function based on bounding box IoU"""
    ann_bbox = ann["bbox"]
    width = ann["width"]
    height = ann["height"]
    kp_ann_bbox = kp_ann["bbox"]

    # ignore instances that are much bigger than the image
    image_area = ann["width"] * ann["height"]
    bbox_area = utils.bbox_area(kp_ann_bbox)
    if bbox_area > 4 * image_area:
        return 0

    # force pose can generate bounding boxes outside the image
    # only match on area inside image
    kp_ann_bbox = utils.bbox_clamp(kp_ann_bbox, width, height)
    return utils.iou(ann_bbox, kp_ann_bbox)


def point_score_fun(ann, kp_ann):
    """Matches annotations based on minimum keypoint distance to a reference point on the human"""

    # ignore instances that are much bigger than the image
    image_area = ann["width"] * ann["height"]
    bbox_area = utils.bbox_area(kp_ann["bbox"])
    if bbox_area > 4 * image_area:
        return (0, 0)

    # ignore instances where the point is outside the bbox
    point = np.array(ann["point"])
    px, py = point
    x, y, w, h = kp_ann["bbox"]
    if (px < x) or (px > x + w) or (py < y) or (py >= y + h):
        return (0, 0)

    keypoints = np.array(kp_ann["keypoint_data"]["keypoints"]).reshape(-1, 3)
    keypoint_score = kp_ann["keypoint_data"]["score"]
    xy = keypoints[keypoints[:, 2] > 0, :2]
    distance = np.sqrt(np.sum((xy - point) ** 2, axis=1))
    if len(distance) > 0:
        # normalize distance by bbox size
        distance /= min(200, max(10, np.sqrt(bbox_area)))
        inv_distance = 1 / min(distance)
    else:
        inv_distance = 0

    result = None

    # no point to match
    if inv_distance == 0:
        result = (0, 0)

    # check if point is on torso
    global _torso_indices
    if (inv_distance > 2) and np.all(keypoints[_torso_indices, 2] > 0):
        torso_keypoints = keypoints[_torso_indices, :2]
        torso = matplotlib.path.Path(torso_keypoints)
        if torso.contains_point(point) and (keypoint_score > 0.05):
            # if there are multiple instances that have the point on the torso then we want to match
            # the one with the highest keypoint score
            # if there are two instances with same score then match the one with the smallest distance
            result = (1e10 * keypoint_score, inv_distance)

    # default case
    if result is None:
        result = (inv_distance, keypoint_score)

    return result


def name_from_output_dir(output_dir):
    """Returns experiment name from experiment output directory"""
    parts = output_dir.split("/")
    return parts[-1] if parts[-1] else parts[-2]
