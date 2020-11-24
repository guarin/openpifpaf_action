from openpifpaf.datasets.constants import COCO_KEYPOINTS


def _index_dict(items):
    return {name: idx for idx, name in enumerate(items)}


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

VCOCO_ACTION_DICT = _index_dict(VCOCO_ACTION_NAMES)

COCO_KEYPOINT_DICT = _index_dict(COCO_KEYPOINTS)

PASCAL_VOC_2012_ACTIONS = [
    "jumping",
    "other",
    "phoning",
    "playinginstrument",
    "reading",
    "ridingbike",
    "ridinghorse",
    "running",
    "takingphoto",
    "usingcomputer",
    "walking",
]

PASCAL_VOC_2012_ACTION_DICT = _index_dict(PASCAL_VOC_2012_ACTIONS)
