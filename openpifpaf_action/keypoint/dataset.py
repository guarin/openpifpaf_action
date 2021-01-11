import torch
import json


def load_annotations(file):
    anns = json.load(open(file))
    anns = [a for a in anns if ("actions" in a) and ("keypoints" in a)]
    return anns


def preprocess(anns, input_preprocess, target_preprocess):
    return anns, input_preprocess(anns), target_preprocess(anns)
