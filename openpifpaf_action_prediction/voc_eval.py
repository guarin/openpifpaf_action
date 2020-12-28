import numpy as np
import json
from xml.etree import ElementTree
import glob
from collections import defaultdict
import matlab.engine
import os
import shutil
import argparse
import pandas as pd
import matplotlib

from openpifpaf.plugins.coco.constants import COCO_KEYPOINTS
from openpifpaf_action_prediction import utils
from openpifpaf_action_prediction import match
from openpifpaf_action_prediction.datasets import voc2012


def load_json(file):
    """Loads an annotation file and filters for action annotations"""
    anns = json.load(open(file))
    filename = file.split("/")[-1][:-17]
    anns = [a for a in anns if "action_probabilities" in a]
    anns = list(sorted(anns, key=lambda a: a["keypoint_data"]["score"], reverse=True))
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
    point = np.array(ann["point"])
    keypoints = np.array(kp_ann["keypoint_data"]["keypoints"]).reshape(-1, 3)
    keypoint_score = kp_ann["keypoint_data"]["score"]
    global _torso_indices
    if np.all(keypoints[_torso_indices, 2] > 0):
        torso_keypoints = keypoints[_torso_indices, :2]
        torso = matplotlib.path.Path(torso_keypoints)
        if torso.contains_point(point):
            return (1e10, keypoint_score)
    xy = keypoints[keypoints[:, 2] > 0, :2]
    distance = np.sqrt(np.sum((xy - point) ** 2, axis=1))
    if len(distance) > 0:
        distance /= max(10, np.sqrt(utils.bbox_area(kp_ann["bbox"])))
        inv_distance = 1 / min(distance)
        return (inv_distance, keypoint_score)
    else:
        return (0, keypoint_score)


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


def build_voc_data(matcher, eval_sets):
    voc_data = defaultdict(list)
    match_counts = defaultdict(lambda: 0)
    matched_eval_sets = defaultdict(dict)
    for eval_ann, pred_ann, _ in matcher.inner_matches():
        for action, prob in zip(
            pred_ann["all_actions"], pred_ann["action_probabilities"]
        ):
            name = eval_ann["filename"][:-4]
            object_id = str(eval_ann["object_id"] + 1)
            if (action in eval_sets) and ((name, object_id) in eval_sets[action]):
                match_counts[action] += 1
                if (prob is not None) and (prob >= 0):
                    voc_data[action].append([name, object_id, str(prob)])
                    matched_eval_sets[action][(name, object_id)] = eval_sets[action][
                        (name, object_id)
                    ]
    return voc_data, match_counts, matched_eval_sets


def write_eval_sets(eval_sets, temp_dir, set_name):
    for action, values in eval_sets.items():
        data = []
        for (filename, object_id), label in values.items():
            data.append([filename, object_id, str(label)])
        np.savetxt(os.path.join(temp_dir, f"{action}_{set_name}.txt"), data, fmt="%s")


def name_from_output_dir(output_dir):
    parts = output_dir.split("/")
    return parts[-1] if parts[-1] else parts[-2]


def eval_all(output_dir, method, threshold, set_dir, anns_dir, voc_devkit_dir):
    """Evaluates all results in an output directory"""
    name = name_from_output_dir(output_dir)
    threshold_str = (
        "0" + str(int(threshold * 10)) if method == "bbox" else "0" + str(threshold[0])
    )

    results = {}
    for set_name in {"train", "val"}:
        prediction_dirs = glob.glob(
            os.path.join(output_dir, f"{set_name}_predictions*")
        )

        anns_file = f"{set_name}_{threshold_str}.json"
        if method == "point":
            anns_file = "point_" + anns_file
        eval_anns_file = os.path.join(anns_dir, anns_file)

        for prediction_dir in sorted(prediction_dirs):
            epoch = prediction_dir.split("_")[-1][-3:]

            print(f"Evaluating {name} {set_name} {epoch}")
            result, matched_result = voc_eval(
                method=method,
                threshold=threshold,
                set_name=set_name,
                files=os.path.join(prediction_dir, "*.json"),
                temp_dir=os.path.join(output_dir, "temp_voc_dir"),
                set_dir=set_dir,
                eval_anns_file=eval_anns_file,
                voc_devkit_dir=voc_devkit_dir,
            )

            results[(name, set_name, epoch, "all")] = result
            results[(name, set_name, epoch, "matched")] = matched_result

    return results


def print_stats(stats, title):
    print(f"-- {title} --")
    for title, stat in stats.items():
        print(f"{title:<31} {stat:>5}")


def voc_eval(
    method,
    threshold,
    set_name,
    files,
    temp_dir,
    set_dir,
    eval_anns_file,
    voc_devkit_dir,
):
    # load predictions
    pred_anns = dict(load_json(file) for file in glob.glob(files))

    # load eval sets per action class
    set_files = os.path.join(set_dir, f"*_{set_name}.txt")
    eval_sets = {
        file.split("/")[-1][: -(len(set_name) + 5)]: np.loadtxt(file, dtype=object)
        for file in glob.glob(set_files)
    }
    eval_sets = {k: {tuple(x[:2]): x[2] for x in v} for k, v in eval_sets.items()}
    eval_counts = {k: len(v) for k, v in eval_sets.items()}

    # load eval annotations
    eval_anns = defaultdict(list)
    for ann in json.load(open(eval_anns_file)):
        if "object_id" in ann:
            eval_anns[ann["filename"]].append(ann)

    # sort annotations by filename
    pred_anns = dict(sorted(pred_anns.items()))
    eval_anns = dict(sorted(eval_anns.items()))

    # match annotations
    matcher = match.ListMatcher(point_score_fun if method == "point" else score_fun)
    matcher.match(eval_anns.values(), pred_anns.values(), threshold=threshold)
    stats = matcher_stats(matcher)

    print_stats(stats, f"{set_name}")

    # write annotations for matlab
    competition = "comp11" if method == "point" else "comp9"
    os.makedirs(temp_dir, exist_ok=True)
    voc_data, match_counts, matched_eval_sets = build_voc_data(matcher, eval_sets)
    for action, data in voc_data.items():
        np.savetxt(
            f"{temp_dir}/{competition}_action_{set_name}_{action}.txt",
            np.array(data, dtype=str),
            fmt="%s",
        )

    write_eval_sets(matched_eval_sets, temp_dir, set_name)

    # get results
    voc_devkit_dir = os.path.abspath(voc_devkit_dir)
    resdir = os.path.abspath(temp_dir)
    clsimgsetpath = os.path.join(os.path.abspath(set_dir), "%s_%s.txt")

    results = matlab_voc_eval(
        voc_devkit_dir=voc_devkit_dir,
        testset=set_name,
        resdir=resdir,
        clsimgsetpath=clsimgsetpath,
        competition=competition,
    )

    clsimgsetpath = os.path.join(os.path.abspath(temp_dir), "%s_%s.txt")
    matched_results = matlab_voc_eval(
        voc_devkit_dir=voc_devkit_dir,
        testset=set_name,
        resdir=resdir,
        clsimgsetpath=clsimgsetpath,
        competition=competition,
    )

    shutil.rmtree(temp_dir)
    return results, matched_results


_matlab_engine = None


def matlab_voc_eval(voc_devkit_dir, testset, resdir, clsimgsetpath, competition):
    """Runs matlab evaluation code"""
    global _matlab_engine
    if _matlab_engine is None:
        _matlab_engine = matlab.engine.start_matlab()

    _matlab_engine.cd(voc_devkit_dir)
    results = _matlab_engine.action_eval_fun(
        testset, resdir, clsimgsetpath, competition
    )
    return list(results[0])


def coco_eval(output_dir):
    files = glob.glob(os.path.join(output_dir, "*.stats.json"))
    name = name_from_output_dir(output_dir)
    results = {}
    for file in sorted(files):
        parts = file.split("/")[-1].split("_")
        set = "_".join(parts[1:-1])
        epoch = parts[-1][-14:-11]
        data = json.load(open(file))
        print(data)
        results[(name, set, epoch)] = data["stats"]
    return results


def main(
    output_dir,
    method,
    threshold,
    set_dir,
    anns_dir,
    voc_devkit_dir,
):
    results = eval_all(
        output_dir=output_dir,
        method=method,
        threshold=threshold,
        set_dir=set_dir,
        anns_dir=anns_dir,
        voc_devkit_dir=voc_devkit_dir,
    )

    columns = ["name", "set", "epoch", "data", "mAP"]
    columns.extend([action for action in voc2012.ACTIONS if action != "other"])
    rows = []
    for key, vals in results.items():
        values = list(key)
        values.extend(vals)
        rows.append(values)
    df = pd.DataFrame(rows, columns=columns)
    df = df.sort_values(["name", "set", "epoch"])
    df.to_csv(os.path.join(output_dir, "voc_results.csv"), index=False)

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

    print("-- VOC --")
    print(df.to_string(index=False))
    print("-- COCO --")
    print(coco_df.to_string(index=False))


parser = argparse.ArgumentParser("VOC Eval")
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--set-dir", type=str, required=True)
parser.add_argument("--anns-dir", type=str, required=True)
parser.add_argument("--voc-devkit-dir", type=str, required=True)
parser.add_argument("--method", default="bbox", type=str)
parser.add_argument("--iou-threshold", default=0.3, type=float)
parser.add_argument("--point-threshold", default=2, type=float)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.method not in {"bbox", "point"}:
        raise ValueError(f"Unknown method: {args.method}")
    threshold = (
        args.iou_threshold if args.method == "bbox" else (args.point_threshold, 0.01)
    )
    main(
        output_dir=args.output_dir,
        method=args.method,
        threshold=threshold,
        set_dir=args.set_dir,
        anns_dir=args.anns_dir,
        voc_devkit_dir=args.voc_devkit_dir,
    )
