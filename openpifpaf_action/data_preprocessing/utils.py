import json
import os
import matplotlib
import numpy as np
import glob

from openpifpaf.plugins.coco.constants import COCO_KEYPOINTS

from openpifpaf_action import utils, match


def filename_from_prediction(file):
    return file.split("/")[-1][:-17]


def load_pifpaf_annotations(file):
    anns = json.load(open(file))
    for ann in anns:
        ann["filename"] = filename_from_prediction(file)
    anns = list(sorted(anns, key=lambda a: a["score"], reverse=True))
    return anns


def ann_files(path, ids):
    ann_files = glob.glob(path)
    ann_files = [a for a in ann_files if a.split("/")[-1][:-4] in ids]
    return ann_files


def keypoint_files(path, ids):
    keypoint_files = glob.glob(path)
    keypoint_files = [
        f for f in keypoint_files if filename_from_prediction(f)[:-4] in ids
    ]
    return keypoint_files


def kp_anns(keypoint_files):
    kp_anns = {
        filename_from_prediction(file): load_pifpaf_annotations(file)
        for file in keypoint_files
    }
    return kp_anns


def set_keypoints_visible(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 3)
    keypoints[keypoints[:, 2] < 0.0001, 2] = 0
    keypoints[keypoints[:, 2] > 0, 2] = 2
    return keypoints.flatten().tolist()


def build_annotation(ann, kp_ann, set_crowd):
    assert ann or kp_ann
    a = {"category_id": 1, "iscrowd": 0}
    if ann:
        a.update(ann)
    if kp_ann:
        a["iscrowd"] = 0
        a["score"] = kp_ann["score"]
        if set_crowd:
            if a["score"] < 0.1:
                a["iscrowd"] = 1
        a["bbox"] = kp_ann["bbox"]
        a["keypoints"] = set_keypoints_visible(kp_ann["keypoints"])
        if ann:
            a["true_bbox"] = ann["bbox"]
        else:
            a["filename"] = kp_ann["filename"]
    return a


def score_fun(ann, kp_ann):
    ann_bbox = ann["bbox"]
    width = ann["width"]
    height = ann["height"]
    kp_ann_bbox = kp_ann["bbox"]

    # ignore instances that are much bigger than the image
    image_area = ann["width"] * ann["height"]
    bbox_area = utils.bbox_area(kp_ann_bbox)
    if bbox_area > 4 * image_area:
        return 0

    # ignore instances with a low score
    if kp_ann["score"] < 0.1:
        return 0

    # force pose can generate bounding boxes outside the image
    kp_ann_bbox = utils.bbox_clamp(kp_ann_bbox, width, height)
    return utils.iou(ann_bbox, kp_ann_bbox)


_coco_dict = utils.index_dict(COCO_KEYPOINTS)
_torso_indices = [
    _coco_dict[name]
    for name in [
        "left_shoulder",
        "left_hip",
        "right_hip",
        "right_shoulder",
        "left_shoulder",
    ]
]


def point_score_fun(ann, kp_ann):
    """Matches annotations based on minimum keypoint distance to a reference point on the human"""

    # ignore instances that are much bigger than the image
    image_area = ann["width"] * ann["height"]
    bbox_area = utils.bbox_area(kp_ann["bbox"])
    if bbox_area > 4 * image_area:
        return (0, 0)

    # ignore instances with a low score
    if kp_ann["score"] < 0.1:
        return (0, 0)

    # ignore instances where the point is outside the bbox
    point = np.array(ann["point"])
    px, py = point
    x, y, w, h = kp_ann["bbox"]
    if (px < x) or (px > x + w) or (py < y) or (py >= y + h):
        return (0, 0)

    keypoints = np.array(kp_ann["keypoints"]).reshape(-1, 3)
    keypoint_score = kp_ann["score"]
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
        if torso.contains_point(point) and (keypoint_score > 0.1):
            # if there are multiple instances that have the point on the torso then we want to match
            # the one with the highest keypoint score
            # if there are two instances with same score then match the one with the smallest distance
            result = (1e10 * keypoint_score, inv_distance)

    # default case
    if result is None:
        result = (inv_distance, keypoint_score)

    return result


def annotations_and_stats(ground_truth, keypoint, score_fun, threshold, set_crowd):
    matcher = match.ListMatcher(score_fun)
    matcher.match(ground_truth, keypoint, threshold=threshold)
    annotations = [
        build_annotation(gt, kp, set_crowd=set_crowd)
        for gt, kp, _ in matcher.outer_matches()
    ]
    stats = match.matcher_stats(matcher)
    return annotations, stats


def print_stats(stats, title):
    print(f"-- {title} --")
    for title, stat in stats.items():
        print(f"{title:<31} {stat:>5}")


def match_anns(anns, kp_anns, train, val, score_fun, threshold, set_crowd):

    train_ground_truth_anns = {k: v for k, v in anns.items() if k[:-4] in train}
    val_ground_truth_anns = {k: v for k, v in anns.items() if k[:-4] in val}
    train_keypoint_anns = {k: v for k, v in kp_anns.items() if k[:-4] in train}
    val_keypoint_anns = {k: v for k, v in kp_anns.items() if k[:-4] in val}

    all_annotations, all_stats = annotations_and_stats(
        anns.values(), kp_anns.values(), score_fun, threshold, set_crowd=set_crowd
    )
    train_annotations, train_stats = annotations_and_stats(
        train_ground_truth_anns.values(),
        train_keypoint_anns.values(),
        score_fun,
        threshold,
        set_crowd=set_crowd,
    )
    val_annotations, val_stats = annotations_and_stats(
        val_ground_truth_anns.values(),
        val_keypoint_anns.values(),
        score_fun,
        threshold,
        set_crowd=set_crowd,
    )

    return (
        all_annotations,
        all_stats,
        train_annotations,
        train_stats,
        val_annotations,
        val_stats,
    )


def plot_anns(ax, file, anns, image_dir):
    colors = list(matplotlib.colors.TABLEAU_COLORS)

    # sort so that colors are the same for different bbox and point sets
    anns = sorted(anns, key=lambda a: tuple(a["bbox"]))

    utils.plot_image(ax, os.path.join(image_dir, file))
    for i, a in enumerate(anns):
        color = colors[i % (len(colors))]
        if "width" in a:
            ax.set_xlim(0, a["width"])
            ax.set_ylim(a["height"], 0)

        if "true_bbox" in a:
            # match
            if a["iscrowd"]:
                utils.plot_keypoints(
                    ax, a["keypoints"], color=color, marker="+", alpha=0.75
                )
            else:
                utils.plot_keypoints(ax, a["keypoints"], color=color, marker=".")
            utils.plot_bbox(ax, a["bbox"], color=color)
            utils.plot_bbox(ax, a["true_bbox"], color="lime")
            if "point" in a:
                point = a["point"]
                ax.scatter([point[0]], [point[1]], color=color, marker="x")
                closest = closest_keypoint(point, a["keypoints"])
                ax.scatter([closest[0]], [closest[1]], color=color, marker="o")

        elif "width" in a:
            # only ground truth
            utils.plot_bbox(ax, a["bbox"], color="lime", linestyle="--")
            if "point" in a:
                point = a["point"]
                ax.scatter([point[0]], [point[1]], color="lime", marker="x")

        else:
            # only keypoint
            # match
            if a["iscrowd"]:
                utils.plot_keypoints(
                    ax, a["keypoints"], color="grey", marker=".", alpha=0.75
                )
                utils.plot_bbox(ax, a["bbox"], color="grey", linewidth=0.75, alpha=0.75)
            else:
                utils.plot_keypoints(ax, a["keypoints"], color=color, marker=".")
                utils.plot_bbox(ax, a["bbox"], color=color, linewidth=0.75)


def closest_keypoint(point, keypoints):
    point = np.array(point)
    keypoints = np.array(keypoints).reshape(-1, 3)
    xy = keypoints[keypoints[:, 2] > 0, :2]
    distance = np.sqrt(np.sum((xy - point) ** 2, axis=1))
    return xy[np.argmin(distance)]
