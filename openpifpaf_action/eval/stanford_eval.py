"""Stanford 40 evaluation module

Usage: python -m openpifpaf_action.eval.stanford_eval --help
"""

import numpy as np
import json
import glob
from collections import defaultdict
import os
import argparse
import pandas as pd
from sklearn.metrics import average_precision_score
from xml.etree import ElementTree

import openpifpaf_action.match
import openpifpaf_action.utils
from openpifpaf_action import eval
from openpifpaf_action import match
from openpifpaf_action.datasets import stanford40


def load_eval_anns(anns_dir):
    all_ann_files = glob.glob(os.path.join(anns_dir, "*.xml"))
    xml_anns = [ElementTree.parse(file) for file in all_ann_files]
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


def eval_all(output_dir, iou_threshold, set_dir, anns_dir, actions):
    """Evaluates all results in an output directory"""
    name = eval.utils.name_from_output_dir(output_dir)
    threshold_str = "0" + str(int(iou_threshold * 10))

    all_anns = load_eval_anns(anns_dir)

    results = {}
    for set_name in ["test"]:
        prediction_dirs = glob.glob(
            os.path.join(output_dir, f"{set_name}_predictions*")
        )

        ids = {
            f[:-4]
            for f in np.loadtxt(os.path.join(set_dir, f"{set_name}.txt"), dtype=str)
        }
        eval_anns = {
            file[:-4]: anns for file, anns in all_anns.items() if file[:-4] in ids
        }

        for prediction_dir in sorted(prediction_dirs):
            epoch = prediction_dir.split("_")[-1][-3:]

            print(f"Evaluating {name} {set_name} {epoch}")
            result, matched_result = stanford_eval(
                iou_threshold=iou_threshold,
                files=os.path.join(prediction_dir, "*.json"),
                eval_anns=eval_anns,
                actions=actions,
            )

            results[(name, set_name, epoch, "all")] = result
            results[(name, set_name, epoch, "matched")] = matched_result

    return results


def stanford_eval(iou_threshold, files, eval_anns, actions):
    # load predictions
    pred_anns = dict(eval.utils.load_json(file) for file in glob.glob(files))

    # load eval annotations
    # eval_anns = defaultdict(list)
    # for ann in json.load(open(eval_anns_file)):
    #    if "object_id" in ann:
    #        eval_anns[ann["filename"]].append(ann)

    # sort annotations by filename
    pred_anns = dict(sorted(pred_anns.items()))
    eval_anns = dict(sorted(eval_anns.items()))

    # match annotations
    matcher = match.ListMatcher(eval.utils.score_fun)
    matcher.match(eval_anns.values(), pred_anns.values(), threshold=iou_threshold)
    stats = openpifpaf_action.match.matcher_stats(matcher)

    # extract predictions
    predictions = defaultdict(list)
    matched_predictions = defaultdict(list)
    for eval_ann, pred_ann, _ in matcher.left_matches():
        for i, action in enumerate(actions):
            target = 1.0 if action in eval_ann["actions"] else 0.0
            if pred_ann is None:
                predictions[action].append((target, 0.0))
            else:
                assert all([a >= 0 for a in pred_ann["action_probabilities"]])
                action_probabilities = np.array(pred_ann["action_probabilities"])
                action_probabilities = action_probabilities / action_probabilities.sum()
                prediction = action_probabilities[i]
                predictions[action].append((target, prediction))
                matched_predictions[action].append((target, prediction))

    results = calculate_aps(predictions)
    matched_results = calculate_aps(matched_predictions)
    return results, matched_results


def calculate_aps(predictions):
    aps = dict()
    for action, scores in predictions.items():
        score_array = np.array(scores)
        y_true = score_array[:, 0]
        y_pred = score_array[:, 1]
        aps[action] = average_precision_score(y_true, y_pred)
    map = sum(aps.values()) / len(aps)
    results = [map]
    results.extend(list(aps.values()))
    return results


def coco_eval(output_dir):
    files = glob.glob(os.path.join(output_dir, "*.stats.json"))
    name = eval.utils.name_from_output_dir(output_dir)
    results = {}
    for file in sorted(files):
        parts = file.split("/")[-1].split("_")
        set = "_".join(parts[1:-1])
        epoch = parts[-1][-14:-11]
        data = json.load(open(file))
        results[(name, set, epoch)] = data["stats"]
    return results


def main(output_dir, iou_threshold, anns_dir, actions, set_dir):
    results = eval_all(
        output_dir=output_dir,
        iou_threshold=iou_threshold,
        anns_dir=anns_dir,
        actions=actions,
        set_dir=set_dir,
    )

    columns = ["name", "set", "epoch", "data", "mAP"]
    columns.extend(actions)
    rows = []
    for key, vals in results.items():
        values = list(key)
        values.extend(vals)
        rows.append(values)
    df = pd.DataFrame(rows, columns=columns)
    df = df.sort_values(["name", "set", "epoch"])
    df.to_csv(os.path.join(output_dir, "stanford_results.csv"), index=False)

    coco_results = coco_eval(output_dir)
    coco_columns = ["name", "set", "epoch"]
    coco_columns.extend(
        ["AP", "AP0.5", "AP0.75", "APM", "APL", "AR", "AR0.5", "AR0.75", "ARM", "ARL"]
    )
    coco_rows = []
    for key, vals in coco_results.items():
        values = list(key)
        values.extend(vals)
        coco_rows.append(values)
    coco_df = pd.DataFrame(coco_rows, columns=coco_columns)
    coco_df.sort_values(["name", "set", "epoch"])

    coco_df.to_csv(os.path.join(output_dir, "coco_results.csv"), index=False)

    print("-- STANFORD --")
    print(df.to_string(index=False))
    print("-- COCO --")
    print(coco_df.to_string(index=False))


parser = argparse.ArgumentParser("Stanford Eval")
parser.add_argument(
    "--output-dir",
    type=str,
    required=True,
    help="Path to directory containing model outputs.",
)
parser.add_argument(
    "--set-dir",
    type=str,
    required=True,
    help="Path to directory containing train/val/test splits.",
)
parser.add_argument(
    "--anns-dir",
    type=str,
    required=True,
    help="Path to ground truth annotations directory.",
)
parser.add_argument(
    "--iou-threshold",
    default=0.3,
    type=float,
    help="Minimum IoU threshold for bbox matching.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        anns_dir=args.anns_dir,
        actions=stanford40.ACTIONS,
        set_dir=args.set_dir,
    )
