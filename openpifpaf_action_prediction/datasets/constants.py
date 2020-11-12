from openpifpaf.datasets.constants import COCO_KEYPOINTS

VCOCO_ACTION_NAMES = [
    "carry",
    "catch",
    "cut",
    "drink",
    "eat",
    "hit",
    "hold",
    "jump",
    "kick",
    "lay",
    "look",
    "point",
    "read",
    "ride",
    "run",
    "sit",
    "skateboard",
    "ski",
    "smile",
    "snowboard",
    "stand",
    "surf",
    "talk_on_phone",
    "throw",
    "walk",
    "work_on_computer",
]

VCOCO_ACTION_DICT = {name: idx for idx, name in enumerate(VCOCO_ACTION_NAMES)}

COCO_KEYPOINT_DICT = {name: idx for idx, name in enumerate(COCO_KEYPOINTS)}
