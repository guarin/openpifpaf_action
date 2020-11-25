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
from openpifpaf.datasets.coco import Coco
from openpifpaf.datasets.collate import (
    collate_images_anns_meta,
    collate_images_targets_meta,
)
from openpifpaf.datasets.constants import (
    COCO_KEYPOINTS,
    COCO_UPRIGHT_POSE,
    COCO_PERSON_SIGMAS,
    COCO_PERSON_SCORE_WEIGHTS,
    COCO_PERSON_SKELETON,
    HFLIP,
)

from openpifpaf_action_prediction import transforms
from openpifpaf_action_prediction import encoder
from openpifpaf_action_prediction import headmeta
from openpifpaf_action_prediction.datasets.constants import (
    PASCAL_VOC_2012_ACTIONS,
    PASCAL_VOC_2012_ACTION_DICT,
)
from openpifpaf_action_prediction import metrics

LOG = logging.getLogger(__name__)

# noinspection PyUnresolvedReferences
class PascalVOC2012(DataModule):

    # cli configurable
    train_annotations = "data/voc2012/train.json"
    val_annotations = "data/voc2012/val.json"
    eval_annotations = val_annotations
    train_image_dir = "data/voc2012/images/"
    val_image_dir = "data/voc2012/images/"
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

    actions = PASCAL_VOC_2012_ACTIONS
    min_actions = 1
    max_actions = len(PASCAL_VOC_2012_ACTIONS)
    required_keypoints = ["left_hip", "right_hip"]
    keypoints = ["left_hip", "right_hip"]
    center = True

    def __init__(self):
        super().__init__()

        cif = openpifpaf.headmeta.Cif(
            name="cif",
            dataset="voc2012",
            keypoints=COCO_KEYPOINTS,
            sigmas=COCO_PERSON_SIGMAS,
            pose=COCO_UPRIGHT_POSE,
            score_weights=COCO_PERSON_SCORE_WEIGHTS,
            draw_skeleton=COCO_PERSON_SKELETON,
        )

        caf = openpifpaf.headmeta.Caf(
            name="caf",
            dataset="voc2012",
            keypoints=COCO_KEYPOINTS,
            sigmas=COCO_PERSON_SIGMAS,
            pose=COCO_UPRIGHT_POSE,
            skeleton=COCO_PERSON_SKELETON,
        )

        aif_center = headmeta.AifCenter(
            name="aif_center",
            dataset="voc2012",
            action_dict=PASCAL_VOC_2012_ACTION_DICT,
            actions=self.actions,
            pose=COCO_UPRIGHT_POSE,
            keypoints=self.keypoints,
        )

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        aif_center.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf, aif_center]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Data Module Pascal VOC 2012")

        group.add_argument("--voc-train-annotations", default=cls.train_annotations)
        group.add_argument("--voc-val-annotations", default=cls.val_annotations)
        group.add_argument("--voc-train-image-dir", default=cls.train_image_dir)
        group.add_argument("--voc-val-image-dir", default=cls.val_image_dir)

        group.add_argument(
            "--voc-square-edge",
            default=cls.square_edge,
            type=int,
            help="square edge of input images",
        )
        assert not cls.extended_scale
        group.add_argument(
            "--voc-extended-scale",
            default=False,
            action="store_true",
            help="augment with an extended scale range",
        )
        group.add_argument(
            "--voc-orientation-invariant",
            default=cls.orientation_invariant,
            type=float,
            help="augment with random orientations",
        )
        group.add_argument(
            "--voc-blur", default=cls.blur, type=float, help="augment with blur"
        )
        group.add_argument(
            "--voc-augmentation",
            default=False,
            action="store_true",
            help="apply data augmentation",
        )
        group.add_argument(
            "--voc-rescale-images",
            default=cls.rescale_images,
            type=float,
            help="overall rescale factor for images",
        )
        group.add_argument(
            "--voc-upsample",
            default=cls.upsample_stride,
            type=int,
            help="head upsample stride",
        )
        group.add_argument(
            "--voc-min-kp-anns",
            default=cls.min_kp_anns,
            type=int,
            help="filter images with fewer keypoint annotations",
        )
        group.add_argument("--voc-bmin", default=cls.bmin, type=float, help="bmin")
        # TODO: better help messages
        group.add_argument(
            "--voc-actions",
            default=cls.actions,
            help="actions",
            nargs="+",
        )
        group.add_argument(
            "--voc-keypoints", default=cls.keypoints, help="keypoints", nargs="+"
        )
        group.add_argument(
            "--voc-min-actions",
            default=cls.min_actions,
            type=int,
            help="minimum number of actions",
        )
        group.add_argument(
            "--voc-max-actions",
            default=len(cls.actions),
            type=int,
            help="maximum number of actions",
        )
        group.add_argument(
            "--voc-required-keypoints", default=cls.required_keypoints, nargs="+"
        )
        group.add_argument(
            "--voc-remove-center",
            default=not cls.center,
            action="store_true",
            help="remove center",
        )

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        cls.train_annotations = args.voc_train_annotations
        cls.val_annotations = args.voc_val_annotations
        cls.train_image_dir = args.voc_train_image_dir
        cls.val_image_dir = args.voc_val_image_dir

        cls.square_edge = args.voc_square_edge
        cls.extended_scale = args.voc_extended_scale
        cls.orientation_invariant = args.voc_orientation_invariant
        cls.blur = args.voc_blur
        cls.augmentation = args.voc_augmentation
        cls.rescale_images = args.voc_rescale_images
        cls.upsample_stride = args.voc_upsample
        cls.min_kp_anns = args.voc_min_kp_anns
        cls.bmin = args.voc_bmin

        cls.actions = args.voc_actions
        cls.min_actions = args.voc_min_actions
        cls.max_actions = args.voc_max_actions
        cls.keypoints = (
            COCO_KEYPOINTS
            if (args.voc_keypoints is not None) and ("all" in args.voc_keypoints)
            else args.voc_keypoints
        )
        cls.required_keypoints = args.voc_required_keypoints
        cls.center = not args.voc_remove_center

    def _encoders(self):
        return [
            encoder.AifCif(self.head_metas[0], bmin=self.bmin),
            encoder.AifCaf(self.head_metas[1], bmin=self.bmin),
            encoder.aif.AifCenter(self.head_metas[2], bmin=self.bmin),
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
        return openpifpaf.transforms.Compose(
            [
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.ToAnnotations(
                    [
                        transforms.annotations.ToAifCenterAnnotations(
                            actions=self.actions,
                            keypoints=self.keypoints,
                            action_dict=PASCAL_VOC_2012_ACTION_DICT,
                        )
                    ]
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
                scale_range=(0.4 * self.rescale_images, 2.0 * self.rescale_images),
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

    # TODO: this should be in __init__ of Coco, should subclass Coco and change accordingly
    def _filter_annotations(self, data: Coco):
        action_indices = [VCOCO_ACTION_DICT[action] for action in self.actions]
        keypoint_indices = [
            COCO_KEYPOINT_DICT[keypoint] for keypoint in self.required_keypoints
        ]
        annotations = []
        images = set()

        for ann in data.coco.dataset["annotations"]:
            num_actions = sum([ann["vcoco_action_labels"][i] for i in action_indices])
            num_keypoints = ann["num_keypoints"]
            if (
                (num_actions >= self.min_actions)
                and (num_actions <= self.max_actions)
                and (num_keypoints >= self.min_kp_anns)
                and all(ann["keypoints"][3 * i + 2] > 0 for i in keypoint_indices)
            ):
                annotations.append(ann)
                images.add(ann["image_id"])

        data.coco.dataset["annotations"] = annotations
        data.coco.dataset["images"] = [
            image for image in data.coco.dataset["images"] if image["id"] in images
        ]
        data.coco.createIndex()
        data.ids = data.coco.getImgIds(catIds=data.category_ids)

        print(f"Annotations: {len(data.coco.anns)}")
        print(f"Images: {len(data.ids)}")

    def train_loader(self):
        train_data = _PascalVOC2012(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
        )
        # self._filter_annotations(train_data)
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
        val_data = _PascalVOC2012(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
        )
        # self._filter_annotations(val_data)
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
        eval_data = _PascalVOC2012(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._eval_preprocess(),
        )
        # self._filter_annotations(eval_data)
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


class _PascalVOC2012(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        image_dir,
        preprocess,
    ):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.preprocess = preprocess

        self.anns = json.load(open(ann_file))
        self.image_files = [a["filename"] for a in self.anns]
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