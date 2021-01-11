"""
Module to produce Stanford 40 training and validation annotations

Usage: python -m openpifpaf_action.data_preprocessing.stanford --help
"""


import numpy as np
import json
from xml.etree import ElementTree
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import argparse

from openpifpaf_action import data_preprocessing
from openpifpaf_action.data_preprocessing.utils import print_stats


def load_anns(ann_files):
    xml_anns = [ElementTree.parse(file) for file in ann_files]
    anns = defaultdict(list)
    for ann in xml_anns:
        filename = ann.find("filename").text
        size = ann.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        objects = ann.findall("object")
        for i, o in enumerate(objects):
            action = o.find("action").text.lower()
            bbox = o.find("bndbox")
            xmax = float(bbox.find("xmax").text)
            xmin = float(bbox.find("xmin").text)
            ymax = float(bbox.find("ymax").text)
            ymin = float(bbox.find("ymin").text)
            anns[filename].append(
                {
                    "filename": filename,
                    "actions": [action],
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "width": width,
                    "height": height,
                    "object_id": i,
                }
            )
    return anns


def save_example_images(image_dir, files, output_dir, train_index, n=100):
    counter = 0
    for file in files:
        filename = file.split("/")[-1] + ".jpg"
        val_anns = train_index[filename]
        if len(val_anns) >= 1:
            fig, ax1 = plt.subplots(ncols=1, figsize=(12, 9))
            data_preprocessing.utils.plot_anns(
                ax1, filename, val_anns, image_dir=image_dir
            )
            fig.savefig(os.path.join(output_dir, filename))
            counter += 1
            if counter >= n:
                break


def main(image_dir, splits_dir, ann_dir, keypoint_dir, output_dir, output_image_dir):
    iou_threshold = 0.3
    set_crowd = True
    train = {
        f[:-4] for f in np.loadtxt(os.path.join(splits_dir, "train.txt"), dtype=str)
    }
    val = {f[:-4] for f in np.loadtxt(os.path.join(splits_dir, "val.txt"), dtype=str)}
    trainval = train.union(val)
    crowd_str = "_crowd" if set_crowd else ""
    threshold_str = str(int(10 * iou_threshold))

    ann_files = data_preprocessing.utils.ann_files(
        os.path.join(ann_dir, "*.xml"), trainval
    )
    keypoint_files = data_preprocessing.utils.keypoint_files(
        os.path.join(keypoint_dir, "*.json"), trainval
    )

    kp_anns = data_preprocessing.utils.kp_anns(keypoint_files)

    anns = load_anns(ann_files)
    anns = dict(sorted(anns.items()))
    kp_anns = dict(sorted(kp_anns.items()))

    (
        trainval_annotations,
        trainval_stats,
        train_annotations,
        train_stats,
        val_annotations,
        val_stats,
    ) = data_preprocessing.utils.match_anns(
        anns,
        kp_anns,
        train,
        val,
        data_preprocessing.utils.score_fun,
        iou_threshold,
        set_crowd=set_crowd,
    )

    print_stats(trainval_stats, "Trainval")
    print_stats(train_stats, "Train")
    print_stats(val_stats, "Val")

    stats = {"trainval": trainval_stats, "train": train_stats, "val": val_stats}

    json.dump(
        trainval_annotations,
        open(
            os.path.join(output_dir, f"trainval_0{threshold_str}{crowd_str}.json"), "w"
        ),
    )
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

    train_index = defaultdict(list)
    for ann in train_annotations:
        train_index[ann["filename"]].append(ann)

    if output_image_dir is not None:
        save_example_images(
            image_dir=image_dir,
            files=train,
            train_index=train_index,
            output_dir=output_image_dir,
        )


parser = argparse.ArgumentParser("Stanford Data Preprocessing")
parser.add_argument(
    "--image-dir", default="data/stanford40/images", type=str, help="Image directory"
)
parser.add_argument(
    "--ann-dir",
    default="data/stanford40/annotations",
    type=str,
    help="Annotations directory",
)
parser.add_argument(
    "--splits-dir",
    default="data/stanford40/my_splits",
    type=str,
    help="Directory containing train.txt and val.txt split files",
)
parser.add_argument(
    "--output-dir", default="data/stanford40", type=str, help="Output directory"
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
