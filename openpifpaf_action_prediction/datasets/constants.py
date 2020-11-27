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

STANFORD_40_ACTIONS = [
    "applauding",
    "blowing_bubbles",
    "brushing_teeth",
    "cleaning_the_floor",
    "climbing",
    "cooking",
    "cutting_trees",
    "cutting_vegetables",
    "drinking",
    "feeding_a_horse",
    "fishing",
    "fixing_a_bike",
    "fixing_a_car",
    "gardening",
    "holding_an_umbrella",
    "jumping",
    "looking_through_a_microscope",
    "looking_through_a_telescope",
    "phoning",
    "playing_guitar",
    "playing_violin",
    "pouring_liquid",
    "pushing_a_cart",
    "reading",
    "riding_a_bike",
    "riding_a_horse",
    "rowing_a_boat",
    "running",
    "shooting_an_arrow",
    "smoking",
    "taking_photos",
    "texting_message",
    "throwing_frisby",
    "using_a_computer",
    "walking_the_dog",
    "washing_dishes",
    "watching_tv",
    "waving_hands",
    "writing_on_a_board",
    "writing_on_a_book",
]

STANFORD_40_ACTION_DICT = _index_dict(STANFORD_40_ACTIONS)
