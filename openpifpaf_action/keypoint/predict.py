import argparse
import json
import torch
import os
import glob
from openpifpaf_action import keypoint
from openpifpaf_action import datasets
from collections import defaultdict


def load_annotations(file):
    anns = json.load(open(file))
    anns = [a for a in anns if ("keypoints" in a)]
    anns_dict = defaultdict(list)
    for a in anns:
        anns_dict[a["filename"]].append(a)
    return anns_dict


def main(checkpoint, image_dir, ann_file, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model = torch.load(checkpoint)
    anns = load_annotations(ann_file)
    files = [f.split("/")[-1] for f in glob.glob(os.path.join(image_dir, "*.jpg"))]
    results = {f: [] for f in files}
    for file, file_anns in anns.items():
        for ann in file_anns:
            new_ann = {"keypoint_data": ann}
            new_ann["bbox"] = ann["bbox"]
            new_ann["all_actions"] = datasets.voc2012.ACTIONS
            bbox = ann["bbox"]
            input = torch.Tensor(ann["keypoints"]).float().reshape(-1, 3)
            input = keypoint.transforms.normalize_with_bbox(input, bbox, scale=True)
            probabilities = model(input.reshape(-1, 3 * 17))[0]
            new_ann["action_probabilities"] = probabilities.tolist()
            results[file].append(new_ann)

    for file, anns in results.items():
        json.dump(anns, open(os.path.join(save_dir, file + ".predictions.json"), "w"))


parser = argparse.ArgumentParser("Keypoint Predict")
parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
parser.add_argument("--ann-file", type=str, required=True, help="Annotation file")
parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
parser.add_argument("--save-dir", type=str, required=True, help="Output directory")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        ann_file=args.ann_file,
        image_dir=args.image_dir,
        save_dir=args.save_dir,
    )
