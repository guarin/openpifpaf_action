import argparse
import json
import torch
from collections import defaultdict
import copy
import os
import PIL
import logging

import openpifpaf
from openpifpaf.datasets.module import DataModule
from openpifpaf.datasets.collate import (
    collate_images_anns_meta,
    collate_images_targets_meta,
)
from openpifpaf.plugins.coco.constants import (
    COCO_KEYPOINTS,
    COCO_UPRIGHT_POSE,
    COCO_PERSON_SIGMAS,
    COCO_PERSON_SCORE_WEIGHTS,
    COCO_PERSON_SKELETON,
    DENSER_COCO_PERSON_CONNECTIONS,
    HFLIP,
)

from openpifpaf_action_prediction import transforms
from openpifpaf_action_prediction import encoder
from openpifpaf_action_prediction import headmeta

from openpifpaf_action_prediction import metrics

LOG = logging.getLogger(__name__)


ACTIONS = [
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


# noinspection PyUnresolvedReferences
class Stanford40(DataModule):

    # cli configurable
    train_annotations = "data/stanford40/train.json"
    val_annotations = "data/stanford40/val.json"
    eval_annotations = val_annotations
    train_image_dir = "data/stanford40/images/"
    val_image_dir = "data/stanford40/images/"
    eval_image_dir = val_image_dir

    square_edge = 385
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = False
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    bmin = 1.0

    eval_annotation_filter = True
    eval_long_edge = 641
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    actions = ACTIONS
    min_actions = 1
    max_actions = len(ACTIONS)
    required_keypoints = ["left_hip", "right_hip"]
    keypoints = [["left_hip", "right_hip"]]

    def __init__(self):
        super().__init__()

        cif = openpifpaf.headmeta.Cif(
            name="cif",
            dataset="stanford40",
            keypoints=COCO_KEYPOINTS,
            sigmas=COCO_PERSON_SIGMAS,
            pose=COCO_UPRIGHT_POSE,
            score_weights=COCO_PERSON_SCORE_WEIGHTS,
            draw_skeleton=COCO_PERSON_SKELETON,
        )

        caf = openpifpaf.headmeta.Caf(
            name="caf",
            dataset="stanford40",
            keypoints=COCO_KEYPOINTS,
            sigmas=COCO_PERSON_SIGMAS,
            pose=COCO_UPRIGHT_POSE,
            skeleton=COCO_PERSON_SKELETON,
        )

        caf25 = openpifpaf.headmeta.Caf(
            name="caf25",
            dataset="stanford40",
            keypoints=COCO_KEYPOINTS,
            sigmas=COCO_PERSON_SIGMAS,
            pose=COCO_UPRIGHT_POSE,
            skeleton=DENSER_COCO_PERSON_CONNECTIONS,
            sparse_skeleton=COCO_PERSON_SKELETON,
            only_in_field_of_view=True,
        )

        aif_center = headmeta.AifCenter(
            name="aif_center",
            dataset="stanford40",
            actions=self.actions,
            pose=COCO_UPRIGHT_POSE,
            center_keypoints=self.keypoints,
        )

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        caf25.upsample_stride = self.upsample_stride
        aif_center.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf, caf25, aif_center]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Data Module Stanford40")

        group.add_argument(
            "--stanford-train-annotations", default=cls.train_annotations
        )
        group.add_argument("--stanford-val-annotations", default=cls.val_annotations)
        group.add_argument("--stanford-train-image-dir", default=cls.train_image_dir)
        group.add_argument("--stanford-val-image-dir", default=cls.val_image_dir)

        group.add_argument(
            "--stanford-square-edge",
            default=cls.square_edge,
            type=int,
            help="square edge of input images",
        )
        assert not cls.extended_scale
        group.add_argument(
            "--stanford-extended-scale",
            default=False,
            action="store_true",
            help="augment with an extended scale range",
        )
        group.add_argument(
            "--stanford-orientation-invariant",
            default=cls.orientation_invariant,
            type=float,
            help="augment with random orientations",
        )
        group.add_argument(
            "--stanford-blur", default=cls.blur, type=float, help="augment with blur"
        )
        group.add_argument(
            "--stanford-augmentation",
            default=False,
            action="store_true",
            help="apply data augmentation",
        )
        group.add_argument(
            "--stanford-rescale-images",
            default=cls.rescale_images,
            type=float,
            help="overall rescale factor for images",
        )
        group.add_argument(
            "--stanford-upsample",
            default=cls.upsample_stride,
            type=int,
            help="head upsample stride",
        )
        group.add_argument(
            "--stanford-min-kp-anns",
            default=cls.min_kp_anns,
            type=int,
            help="filter images with fewer keypoint annotations",
        )
        group.add_argument("--stanford-bmin", default=cls.bmin, type=float, help="bmin")
        # TODO: better help messages
        group.add_argument(
            "--stanford-actions",
            default=cls.actions,
            help="actions",
            nargs="+",
        )
        group.add_argument(
            "--stanford-keypoints",
            default=cls.keypoints,
            help="keypoints",
            nargs="+",
            type=lambda kps: kps.split(","),
        )
        group.add_argument(
            "--stanford-min-actions",
            default=cls.min_actions,
            type=int,
            help="minimum number of actions",
        )
        group.add_argument(
            "--stanford-max-actions",
            default=len(cls.actions),
            type=int,
            help="maximum number of actions",
        )
        group.add_argument(
            "--stanford-required-keypoints", default=cls.required_keypoints, nargs="+"
        )

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        cls.train_annotations = args.stanford_train_annotations
        cls.val_annotations = args.stanford_val_annotations
        cls.train_image_dir = args.stanford_train_image_dir
        cls.val_image_dir = args.stanford_val_image_dir

        cls.square_edge = args.stanford_square_edge
        cls.extended_scale = args.stanford_extended_scale
        cls.orientation_invariant = args.stanford_orientation_invariant
        cls.blur = args.stanford_blur
        cls.augmentation = args.stanford_augmentation
        cls.rescale_images = args.stanford_rescale_images
        cls.upsample_stride = args.stanford_upsample
        cls.min_kp_anns = args.stanford_min_kp_anns
        cls.bmin = args.stanford_bmin

        cls.actions = args.stanford_actions
        cls.min_actions = args.stanford_min_actions
        cls.max_actions = args.stanford_max_actions
        cls.keypoints = (
            args.stanford_keypoints if args.stanford_keypoints else cls.keypoints
        )
        cls.required_keypoints = args.stanford_required_keypoints

    def _encoders(self):
        return [
            encoder.aif.AifCif(self.head_metas[0], bmin=self.bmin),
            encoder.aif.AifCaf(self.head_metas[1], bmin=self.bmin),
            encoder.aif.AifCaf(self.head_metas[2], bmin=self.bmin),
            encoder.aif.AifCenter(self.head_metas[3], bmin=self.bmin),
        ]

    def _preprocess_no_agumentation(self):
        return openpifpaf.transforms.Compose(
            [
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(self._encoders()),
            ]
        )

    def _eval_preprocess(self):
        to_kp_annotations = openpifpaf.transforms.ToKpAnnotations(
            COCO_CATEGORIES,
            keypoints_by_category={1: self.head_metas[0].keypoints},
            skeleton_by_category={1: self.head_metas[1].skeleton},
        )
        to_aif_center_annotations = transforms.annotations.ToAifCenterAnnotations(
            to_kp_annotations=to_kp_annotations,
            actions=self.actions,
            keypoints=self.keypoints,
            all_keypoints=COCO_KEYPOINTS,
        )

        return openpifpaf.transforms.Compose(
            [
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.ToAnnotations(
                    [to_kp_annotations, to_aif_center_annotations]
                ),
                openpifpaf.transforms.EVAL_TRANSFORM,
            ]
        )

    def _preprocess(self):
        # TODO: Transforms is not in __init__ of pifpaf
        if not self.augmentation:
            return self._preprocess_no_agumentation()

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.25 * self.rescale_images, 2.0 * self.rescale_images),
                power_law=True,
                stretch_range=(0.75, 1.33),
            )
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.8 * self.rescale_images, 2.0 * self.rescale_images),
                power_law=True,
                stretch_range=(0.75, 1.33),
            )

        return openpifpaf.transforms.Compose(
            [
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RandomApply(
                    openpifpaf.transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5
                ),
                rescale_t,
                openpifpaf.transforms.RandomApply(
                    openpifpaf.transforms.Blur(), self.blur
                ),
                openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.RandomApply(
                    openpifpaf.transforms.RotateBy90(), self.orientation_invariant
                ),
                openpifpaf.transforms.TRAIN_TRANSFORM,
                openpifpaf.transforms.Encoders(self._encoders()),
            ]
        )

    def train_loader(self):
        train_data = _Stanford40(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
        )
        return torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=collate_images_targets_meta,
        )

    def val_loader(self):
        val_data = _Stanford40(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
        )
        return torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=collate_images_targets_meta,
        )

    def eval_loader(self):
        eval_data = _Stanford40(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._eval_preprocess(),
        )
        return torch.utils.data.DataLoader(
            eval_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=collate_images_anns_meta,
        )

    def metrics(self):
        return [metrics.pascal_voc_2012.PascalVOC2012(self.actions)]


class _Stanford40(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        image_dir,
        preprocess,
    ):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.preprocess = preprocess

        self.anns = [a for a in json.load(open(ann_file)) if "keypoints" in a]
        self.image_files = list({a["filename"] for a in self.anns})
        self._ann_index = defaultdict(list)
        for a in self.anns:
            self._ann_index[a["filename"]].append(a)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = self.image_files[index]
        anns = copy.deepcopy(self._ann_index[filename])
        image_id = filename[:-4]
        local_file_path = os.path.join(self.image_dir, filename)

        meta = {
            "dataset_index": index,
            "image_id": image_id,
            "file_name": filename,
            "local_file_path": local_file_path,
        }

        with open(local_file_path, "rb") as f:
            image = PIL.Image.open(f).convert("RGB")

        image, anns, meta = self.preprocess(image, anns, meta)

        LOG.debug(meta)
        return image, anns, meta
