"""
Module to produce Pascal VOC 2012 training and validation annotations

Usage: python -m openpifpaf_action.data_preprocessing.pascal --help
"""

import numpy as np
import json
from xml.etree import ElementTree
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import argparse

from openpifpaf_action import data_preprocessing
from openpifpaf_action.data_preprocessing.utils import (
    point_score_fun,
    match_anns,
    plot_anns,
    print_stats,
)


def load_anns(ann_files):
    xml_anns = [ElementTree.parse(file) for file in ann_files]
    anns = defaultdict(list)
    for ann in xml_anns:
        filename = ann.find("filename").text
        objects = ann.findall("object")
        size = ann.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        for object_id, o in enumerate(objects):
            actions = o.find("actions")
            actions = [(a.tag, int(a.text)) for a in actions]
            actions = list(sorted(actions, key=lambda x: x[0]))
            bbox = o.find("bndbox")
            xmax = float(bbox.find("xmax").text)
            xmin = float(bbox.find("xmin").text)
            ymax = float(bbox.find("ymax").text)
            ymin = float(bbox.find("ymin").text)
            difficult = o.find("difficult").text
            point = o.find("point")
            x = float(point.find("x").text)
            y = float(point.find("y").text)
            anns[filename].append(
                {
                    "filename": filename,
                    "actions": [a[0] for a in actions if a[1] == 1],
                    "action_labels": [a[1] for a in actions],
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "difficult": difficult,
                    "object_id": object_id,
                    "width": width,
                    "height": height,
                    "point": [x, y],
                }
            )
    return anns


def save_example_images(
    image_dir, files, output_dir, train_index, point_train_index=None, n=100
):
    counter = 0
    for file in files:
        filename = file.split("/")[-1] + ".jpg"
        val_anns = train_index[filename]
        point_val_anns = point_train_index[filename]
        if len(val_anns) >= 1:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 9))
            plot_anns(ax1, filename, val_anns, image_dir=image_dir)
            plot_anns(ax2, filename, point_val_anns, image_dir=image_dir)
            fig.savefig(os.path.join(output_dir, filename))
            counter += 1
            if counter >= n:
                break


def main(image_dir, splits_dir, ann_dir, keypoint_dir, output_dir, output_image_dir):
    iou_threshold = 0.3
    point_threshold = 2
    set_crowd = True
    train = np.loadtxt(os.path.join(splits_dir, "train.txt"), dtype=str)
    val = np.loadtxt(os.path.join(splits_dir, "val.txt"), dtype=str)
    ids = set(train).union(set(val))
    crowd_str = "_crowd" if set_crowd else "_nocrowd"
    threshold_str = str(int(10 * iou_threshold))
    point_threshold_str = str(point_threshold)

    ann_files = data_preprocessing.utils.ann_files(os.path.join(ann_dir, "*.xml"), ids)
    keypoint_files = data_preprocessing.utils.keypoint_files(
        os.path.join(keypoint_dir, "*.json"), ids
    )

    kp_anns = data_preprocessing.utils.kp_anns(keypoint_files)
    anns = load_anns(ann_files)
    anns = dict(sorted(anns.items()))
    kp_anns = dict(sorted(kp_anns.items()))

    (
        all_annotations,
        all_stats,
        train_annotations,
        train_stats,
        val_annotations,
        val_stats,
    ) = match_anns(
        anns,
        kp_anns,
        train,
        val,
        data_preprocessing.utils.score_fun,
        iou_threshold,
        set_crowd=set_crowd,
    )

    (
        point_all_annotations,
        point_all_stats,
        point_train_annotations,
        point_train_stats,
        point_val_annotations,
        point_val_stats,
    ) = match_anns(
        anns,
        kp_anns,
        train,
        val,
        point_score_fun,
        (point_threshold, 0.1),
        set_crowd=set_crowd,
    )

    print_stats(all_stats, "All")
    print_stats(train_stats, "Train")
    print_stats(val_stats, "Val")
    print_stats(point_all_stats, "All")
    print_stats(point_train_stats, "Train")
    print_stats(point_val_stats, "Val")

    stats = {"all": all_stats, "train": train_stats, "val": val_stats}
    point_stats = {
        "all": point_all_stats,
        "train": point_train_stats,
        "val": point_val_stats,
    }

    json.dump(
        train_annotations,
        open(os.path.join(output_dir, f"train_0{threshold_str}{crowd_str}.json"), "w"),
    )
    json.dump(
        val_annotations,
        open(os.path.join(output_dir, f"val_0{threshold_str}{crowd_str}.json"), "w"),
    )
    json.dump(
        stats,
        open(os.path.join(output_dir, f"stats_0{threshold_str}{crowd_str}.json"), "w"),
    )
    json.dump(
        point_train_annotations,
        open(
            os.path.join(
                output_dir, f"point_train_0{point_threshold_str}{crowd_str}.json"
            ),
            "w",
        ),
    )
    json.dump(
        point_val_annotations,
        open(
            os.path.join(
                output_dir,
                f"point_val_0{point_threshold_str}{crowd_str}.json",
            ),
            "w",
        ),
    )
    json.dump(
        point_stats,
        open(
            os.path.join(
                output_dir,
                f"point_stats_0{point_threshold_str}{crowd_str}.json",
            ),
            "w",
        ),
    )

    train_index = defaultdict(list)
    for ann in train_annotations:
        train_index[ann["filename"]].append(ann)

    point_train_index = defaultdict(list)
    for ann in point_train_annotations:
        point_train_index[ann["filename"]].append(ann)

    if output_image_dir is not None:
        save_example_images(
            image_dir=image_dir,
            files=train,
            output_dir=output_image_dir,
            train_index=train_index,
            point_train_index=point_train_index,
        )


parser = argparse.ArgumentParser("Pascal VOC Data Preprocessing")
parser.add_argument(
    "--image-dir", default="data/voc2012/images", type=str, help="Image directory"
)
parser.add_argument(
    "--ann-dir",
    default="data/voc2012/annotations",
    type=str,
    help="Annotations directory",
)
parser.add_argument(
    "--splits-dir",
    default="data/voc2012/image_sets/action",
    type=str,
    help="Directory containing train.txt and val.txt split files",
)
parser.add_argument(
    "--output-dir", default="data/voc2012", type=str, help="Output directory"
)
parser.add_argument(
    "--keypoint-dir",
    type=str,
    help="Directory with pifpaf keypoint prediction json files for images in image-dir",
    required=True,
)
parser.add_argument(
    "--output-image-dir",
    type=str,
    help="Directory in which to save images with match indications",
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        image_dir=args.image_dir,
        splits_dir=args.splits_dir,
        ann_dir=args.ann_dir,
        keypoint_dir=args.keypoint_dir,
        output_dir=args.output_dir,
        output_image_dir=args.output_image_dir,
    )
