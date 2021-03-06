"""Pascal VOC evaluation module

Usage: python -m openpifpaf_action.eval.pascal_eval --help
"""

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

from openpifpaf.plugins.coco.constants import COCO_KEYPOINTS

import openpifpaf_action.match
import openpifpaf_action.utils
from openpifpaf_action import utils
from openpifpaf_action import match
from openpifpaf_action.datasets import voc2012
from openpifpaf_action import eval


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


def load_eval_anns(anns_dir):
    all_ann_files = glob.glob(os.path.join(anns_dir, "*.xml"))
    xml_anns = [ElementTree.parse(file) for file in all_ann_files]
    anns = defaultdict(list)
    for ann in xml_anns:
        filename = ann.find("filename").text
        objects = ann.findall("object")
        size = ann.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        for object_id, o in enumerate(objects):
            actions = o.find("actions")
            if actions:
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


def eval_all(output_dir, method, threshold, set_dir, anns_dir, voc_devkit_dir):
    """Evaluates all results in an output directory"""
    name = eval.utils.name_from_output_dir(output_dir)
    threshold_str = (
        "0" + str(int(threshold * 10)) if method == "bbox" else "0" + str(threshold[0])
    )

    all_anns = load_eval_anns(anns_dir)

    results = {}
    match_results = {}
    for set_name in {"train", "val"}:
        prediction_dirs = glob.glob(
            os.path.join(output_dir, f"{set_name}_predictions*")
        )

        ids = set(np.loadtxt(os.path.join(set_dir, f"{set_name}.txt"), dtype=str))
        eval_anns = {
            file[:-4]: anns for file, anns in all_anns.items() if file[:-4] in ids
        }

        for prediction_dir in sorted(prediction_dirs):
            epoch = prediction_dir.split("_")[-1][-3:]
            pred_anns = dict(
                eval.utils.load_json(file)
                for file in glob.glob(os.path.join(prediction_dir, "*.json"))
            )

            print(f"Evaluating {name} {set_name} {epoch}")
            result, matched_result, matcher = voc_eval(
                method=method,
                threshold=threshold,
                set_name=set_name,
                pred_anns=pred_anns,
                temp_dir=os.path.join(output_dir, "temp_voc_dir"),
                set_dir=set_dir,
                eval_anns=eval_anns,
                voc_devkit_dir=voc_devkit_dir,
                return_matcher=True,
            )
            results[(name, set_name, epoch, "all")] = result
            results[(name, set_name, epoch, "matched")] = matched_result
            stats = openpifpaf_action.match.matcher_stats(matcher)
            match_results[(name, set_name, epoch)] = stats

    return results, match_results


def print_stats(stats, title):
    print(f"-- {title} --")
    for title, stat in stats.items():
        print(f"{title:<31} {stat:>5}")


def voc_eval(
    method,
    threshold,
    set_name,
    pred_anns,
    temp_dir,
    set_dir,
    eval_anns,
    voc_devkit_dir,
    return_matcher=False,
):
    # load eval sets per action class
    set_files = os.path.join(set_dir, f"*_{set_name}.txt")
    eval_sets = {
        file.split("/")[-1][: -(len(set_name) + 5)]: np.loadtxt(file, dtype=object)
        for file in glob.glob(set_files)
    }
    eval_sets = {k: {tuple(x[:2]): x[2] for x in v} for k, v in eval_sets.items()}
    eval_counts = {k: len(v) for k, v in eval_sets.items()}

    # sort annotations by filename
    pred_anns = dict(sorted(pred_anns.items()))
    eval_anns = dict(sorted(eval_anns.items()))

    # match annotations
    matcher = match.ListMatcher(
        eval.utils.point_score_fun if method == "point" else eval.utils.score_fun
    )
    matcher.match(eval_anns.values(), pred_anns.values(), threshold=threshold)

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
    if return_matcher:
        return results, matched_results, matcher
    else:
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
    name = eval.utils.name_from_output_dir(output_dir)
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
    results, match_results = eval_all(
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

    match_columns = ["name", "set", "epoch"]
    match_columns.extend(
        [
            "before_total",
            "before_ann",
            "before_kp",
            "after_total",
            "after_ann_matched",
            "after_ann_matched %",
            "after_ann_action",
            "after_ann_action %",
            "after_unmatched_ann",
            "after_unmatched_kp",
        ]
    )
    match_rows = []
    for key, vals in match_results.items():
        values = list(key)
        values.extend(list(vals.values()))
        match_rows.append(values)
    match_df = pd.DataFrame(match_rows, columns=match_columns)
    match_df.sort_values(["name", "set", "epoch"])
    match_df.to_csv(os.path.join(output_dir, "match_results.csv"), index=False)

    print("-- VOC --")
    print(df.to_string(index=False))
    print("-- COCO --")
    print(coco_df.to_string(index=False))
    print("-- MATCH --")
    print(match_df.to_string(index=False))


parser = argparse.ArgumentParser("VOC Eval")
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
    "--voc-devkit-dir", type=str, required=True, help="Path to matlab devkit directory."
)
parser.add_argument(
    "--method",
    default="bbox",
    type=str,
    help="Pascal matching protocol. One of: bbox, point",
)
parser.add_argument(
    "--iou-threshold",
    default=0.3,
    type=float,
    help="Minimum IoU threshold for bbox matching.",
)
parser.add_argument(
    "--point-threshold",
    default=1,
    type=float,
    help="Minimum threshold for point matching protocol.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.method not in {"bbox", "point"}:
        raise ValueError(f"Unknown method: {args.method}")
    threshold = (
        args.iou_threshold if args.method == "bbox" else (args.point_threshold, 0.1)
    )
    main(
        output_dir=args.output_dir,
        method=args.method,
        threshold=threshold,
        set_dir=args.set_dir,
        anns_dir=args.anns_dir,
        voc_devkit_dir=args.voc_devkit_dir,
    )
