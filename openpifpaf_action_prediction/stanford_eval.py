import numpy as np
import json
import glob
from collections import defaultdict
import os
import shutil
import argparse
import pandas as pd
from sklearn.metrics import average_precision_score

from openpifpaf_action_prediction import utils
from openpifpaf_action_prediction import match
from openpifpaf_action_prediction.datasets import stanford40


def load_json(file):
    """Loads an annotation file and filters for action annotations"""
    anns = json.load(open(file))
    filename = file.split("/")[-1][:-17]
    anns = [
        a
        for a in anns
        if "action_probabilities" in a and all(a["action_probabilities"])
    ]
    return filename, anns


def score_fun(ann, kp_ann):
    ann_bbox = ann["bbox"]
    width = ann["width"]
    height = ann["height"]
    kp_ann_bbox = kp_ann["bbox"]
    # force pose can generate bounding boxes outside the image
    # only match on area inside image
    kp_ann_bbox = utils.bbox_clamp(kp_ann_bbox, width, height)
    return utils.iou(ann_bbox, kp_ann_bbox)


def matcher_stats(matcher):
    num_annotations, num_ground_truth, num_keypoint, num_matched = matcher.counts()
    before_num_ground_truth = len(matcher.left_matches())
    before_num_keypoint = len(matcher.right_matches())
    before_num_annotations = before_num_ground_truth + before_num_keypoint
    values = [
        before_num_annotations,
        before_num_ground_truth,
        before_num_keypoint,
        num_annotations,
        num_matched,
        num_matched / num_ground_truth,
        num_ground_truth - num_matched,
        num_keypoint - num_matched,
    ]
    titles = [
        "Before Total",
        "Before Annotated",
        "Before Keypoint",
        "After Total",
        "After Matched",
        "After Annotated Matched %",
        "After Unmatched Annotated",
        "After Unmatched Keypoint",
    ]
    stats = dict(zip(titles, values))
    return stats


def name_from_output_dir(output_dir):
    parts = output_dir.split("/")
    return parts[-1] if parts[-1] else parts[-2]


def eval_all(output_dir, iou_threshold, anns_dir, actions):
    """Evaluates all results in an output directory"""
    name = name_from_output_dir(output_dir)
    threshold_str = "0" + str(int(iou_threshold * 10))

    results = {}
    for set_name in ["train", "val"]:
        prediction_dirs = glob.glob(
            os.path.join(output_dir, f"{set_name}_predictions*")
        )
        for prediction_dir in sorted(prediction_dirs):
            epoch = prediction_dir.split("_")[-1][-3:]

            print(f"Evaluating {name} {set_name} {epoch}")
            result, matched_result = stanford_eval(
                iou_threshold=iou_threshold,
                files=os.path.join(prediction_dir, "*.json"),
                eval_anns_file=os.path.join(
                    anns_dir, f"{set_name}_{threshold_str}.json"
                ),
                actions=actions,
            )

            results[(name, set_name, epoch, "all")] = result
            results[(name, set_name, epoch, "matched")] = matched_result

    return results


def stanford_eval(iou_threshold, files, eval_anns_file, actions):
    # load predictions
    pred_anns = dict(load_json(file) for file in glob.glob(files))

    # load eval annotations
    eval_anns = defaultdict(list)
    for ann in json.load(open(eval_anns_file)):
        if "object_id" in ann:
            eval_anns[ann["filename"]].append(ann)

    # sort annotations by filename
    pred_anns = dict(sorted(pred_anns.items()))
    eval_anns = dict(sorted(eval_anns.items()))

    print(len(pred_anns))
    print(len(eval_anns))

    # match annotations
    matcher = match.ListMatcher(score_fun)
    matcher.match(eval_anns.values(), pred_anns.values(), threshold=iou_threshold)
    stats = matcher_stats(matcher)

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
        print("Target ", min(y_true), max(y_true))
        print("Prediction ", min(y_pred), max(y_pred))
        aps[action] = average_precision_score(y_true, y_pred)
    map = sum(aps.values()) / len(aps)
    results = [map]
    results.extend(list(aps.values()))
    return results


def coco_eval(output_dir):
    files = glob.glob(os.path.join(output_dir, "*.stats.json"))
    name = name_from_output_dir(output_dir)
    results = {}
    for file in sorted(files):
        parts = file.split("/")[-1].split("_")
        set = "_".join(parts[1:-1])
        epoch = parts[-1][-14:-11]
        data = json.load(open(file))
        results[(name, set, epoch)] = data["stats"]
    return results


def main(output_dir, iou_threshold, anns_dir, actions):
    results = eval_all(
        output_dir=output_dir,
        iou_threshold=iou_threshold,
        anns_dir=anns_dir,
        actions=actions,
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
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--anns-dir", type=str, required=True)
parser.add_argument("--iou-threshold", default=0.3, type=float)
parser.add_argument("--actions", default=[], nargs="+")


if __name__ == "__main__":
    args = parser.parse_args()
    actions = args.actions if args.actions else stanford40.ACTIONS
    main(
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        anns_dir=args.anns_dir,
        actions=actions,
    )
