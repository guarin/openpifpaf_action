import argparse

import torch

import openpifpaf
from openpifpaf.datasets.module import DataModule
from openpifpaf.datasets.coco import Coco
from openpifpaf.datasets.collate import (
    collate_images_anns_meta,
    collate_images_targets_meta,
)
from openpifpaf.datasets.constants import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    COCO_PERSON_SKELETON,
    COCO_PERSON_SIGMAS,
    COCO_PERSON_SCORE_WEIGHTS,
    COCO_UPRIGHT_POSE,
    DENSER_COCO_PERSON_CONNECTIONS,
    HFLIP,
)

from openpifpaf_action_prediction import encoder
from openpifpaf_action_prediction import headmeta
from openpifpaf_action_prediction.datasets.constants import (
    VCOCO_ACTION_NAMES,
    VCOCO_ACTION_DICT,
)


try:
    import pycocotools.coco

    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass


class Vcoco(DataModule):
    _test2017_annotations = "data-mscoco/annotations/image_info_test2017.json"
    _testdev2017_annotations = "data-mscoco/annotations/image_info_test-dev2017.json"
    _test2017_image_dir = "data-mscoco/images/test2017/"

    # cli configurable
    train_annotations = "generated_data/vcoco/vcoco_train.json"
    val_annotations = "generated_data/vcoco/vcoco_train.json"
    eval_annotations = val_annotations
    train_image_dir = "data/coco/train2014/"
    val_image_dir = "data/coco/train2014/"
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

    actions = VCOCO_ACTION_NAMES
    min_actions = 1
    max_actions = len(VCOCO_ACTION_NAMES)
    keypoints = None
    center = True

    def __init__(self):
        super().__init__()

        print(self.actions)
        print("-" * 100)
        print(self.center)

        aif = headmeta.Aif(
            "aif",
            "vcoco",
            actions=self.actions,
            keypoints=self.keypoints,
            center=self.center,
            keypoint_sigmas=[],
            center_sigma=1,
            pose=COCO_UPRIGHT_POSE,
        )
        aif.upsample_stride = self.upsample_stride
        self.head_metas = [aif]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group("data module vcoco")

        group.add_argument("--vcoco-train-annotations", default=cls.train_annotations)
        group.add_argument("--vcoco-val-annotations", default=cls.val_annotations)
        group.add_argument("--coco-train-image-dir", default=cls.train_image_dir)
        group.add_argument("--coco-val-image-dir", default=cls.val_image_dir)

        group.add_argument(
            "--coco-square-edge",
            default=cls.square_edge,
            type=int,
            help="square edge of input images",
        )
        assert not cls.extended_scale
        group.add_argument(
            "--coco-extended-scale",
            default=False,
            action="store_true",
            help="augment with an extended scale range",
        )
        group.add_argument(
            "--coco-orientation-invariant",
            default=cls.orientation_invariant,
            type=float,
            help="augment with random orientations",
        )
        group.add_argument(
            "--coco-blur", default=cls.blur, type=float, help="augment with blur"
        )
        group.add_argument(
            "--coco-augmentation",
            dest="coco_augmentation",
            default=False,
            action="store_true",
            help="apply data augmentation",
        )
        group.add_argument(
            "--coco-rescale-images",
            default=cls.rescale_images,
            type=float,
            help="overall rescale factor for images",
        )
        group.add_argument(
            "--coco-upsample",
            default=cls.upsample_stride,
            type=int,
            help="head upsample stride",
        )
        group.add_argument(
            "--coco-min-kp-anns",
            default=cls.min_kp_anns,
            type=int,
            help="filter images with fewer keypoint annotations",
        )
        group.add_argument("--coco-bmin", default=cls.bmin, type=float, help="bmin")
        # TODO: better help messages
        group.add_argument(
            "--actions",
            default=cls.actions,
            help="actions",
            nargs="+",
        )
        group.add_argument(
            "--keypoints", default=cls.keypoints, help="keypoints", nargs="+"
        )
        group.add_argument(
            "--min-actions",
            default=cls.min_actions,
            type=int,
            help="minimum number of actions",
        )
        group.add_argument(
            "--max-actions",
            default=len(cls.actions),
            type=int,
            help="maximum number of actions",
        )
        group.add_argument(
            "--remove-center",
            default=not cls.center,
            action="store_true",
            help="remove center",
        )

        # evaluation
        # eval_set_group = group.add_mutually_exclusive_group()
        # eval_set_group.add_argument(
        #     "--cocokp-eval-test2017", default=False, action="store_true"
        # )
        # eval_set_group.add_argument(
        #     "--cocokp-eval-testdev2017", default=False, action="store_true"
        # )
        #
        # assert cls.eval_annotation_filter
        # group.add_argument(
        #     "--coco-no-eval-annotation-filter",
        #     dest="coco_eval_annotation_filter",
        #     default=True,
        #     action="store_false",
        # )
        # group.add_argument(
        #     "--coco-eval-long-edge",
        #     default=cls.eval_long_edge,
        #     type=int,
        #     help="set to zero to deactivate rescaling",
        # )
        # assert not cls.eval_extended_scale
        # group.add_argument(
        #     "--coco-eval-extended-scale", default=False, action="store_true"
        # )
        # group.add_argument(
        #     "--coco-eval-orientation-invariant",
        #     default=cls.eval_orientation_invariant,
        #     type=float,
        # )

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # vcoco/coco specific
        cls.train_annotations = args.vcoco_train_annotations
        cls.val_annotations = args.vcoco_val_annotations
        cls.train_image_dir = args.coco_train_image_dir
        cls.val_image_dir = args.coco_val_image_dir

        cls.square_edge = args.coco_square_edge
        cls.extended_scale = args.coco_extended_scale
        cls.orientation_invariant = args.coco_orientation_invariant
        cls.blur = args.coco_blur
        cls.augmentation = args.coco_augmentation
        cls.rescale_images = args.coco_rescale_images
        cls.upsample_stride = args.coco_upsample
        cls.min_kp_anns = args.coco_min_kp_anns
        cls.bmin = args.coco_bmin

        cls.actions = args.actions
        cls.min_actions = args.min_actions
        cls.max_actions = args.max_actions
        cls.keypoints = (
            COCO_KEYPOINTS
            if (args.keypoints is not None) and ("all" in args.keypoints)
            else args.keypoints
        )
        cls.center = not args.remove_center

        # evaluation
        # cls.eval_annotation_filter = args.coco_eval_annotation_filter
        # if args.cocokp_eval_test2017:
        #     cls.eval_image_dir = cls._test2017_image_dir
        #     cls.eval_annotations = cls._test2017_annotations
        #     cls.annotation_filter = False
        # if args.cocokp_eval_testdev2017:
        #     cls.eval_image_dir = cls._test2017_image_dir
        #     cls.eval_annotations = cls._testdev2017_annotations
        #     cls.annotation_filter = False
        # cls.eval_long_edge = args.coco_eval_long_edge
        # cls.eval_orientation_invariant = args.coco_eval_orientation_invariant
        # cls.eval_extended_scale = args.coco_eval_extended_scale
        #
        # if (
        #     (args.cocokp_eval_test2017 or args.cocokp_eval_testdev2017)
        #     and not args.write_predictions
        #     and not args.debug
        # ):
        #     raise Exception("have to use --write-predictions for this dataset")

    def _preprocess(self):
        encoders = [encoder.aif.Aif(self.head_metas[0], bmin=self.bmin)]

        # TODO: Transforms is not in __init__ of pifpaf
        if not self.augmentation:
            return openpifpaf.transforms.Compose(
                [
                    openpifpaf.transforms.NormalizeAnnotations(),
                    openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                    openpifpaf.transforms.CenterPad(self.square_edge),
                    openpifpaf.transforms.EVAL_TRANSFORM,
                    openpifpaf.transforms.Encoders(encoders),
                ]
            )

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
                openpifpaf.transforms.Encoders(encoders),
            ]
        )

    # TODO: this should be in __init__ of Coco, should subclass Coco and change accordingly
    def _filter_annotations(self, data: Coco):
        action_indices = [VCOCO_ACTION_DICT[action] for action in self.actions]
        annotations = []
        images = set()

        for ann in data.coco.dataset["annotations"]:
            num_actions = sum([ann["vcoco_action_labels"][i] for i in action_indices])
            num_keypoints = ann["num_keypoints"]
            if (
                (num_actions >= self.min_actions)
                and (num_actions <= self.max_actions)
                and (num_keypoints >= self.min_kp_anns)
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
        train_data = Coco(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        self._filter_annotations(train_data)
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
        val_data = Coco(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        self._filter_annotations(val_data)
        return torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=collate_images_targets_meta,
        )

    # @classmethod
    # def common_eval_preprocess(cls):
    #     rescale_t = None
    #     if cls.eval_extended_scale:
    #         assert cls.eval_long_edge
    #         rescale_t = [
    #             transforms.DeterministicEqualChoice(
    #                 [
    #                     transforms.RescaleAbsolute(cls.eval_long_edge),
    #                     transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
    #                 ],
    #                 salt=1,
    #             )
    #         ]
    #     elif cls.eval_long_edge:
    #         rescale_t = transforms.RescaleAbsolute(cls.eval_long_edge)
    #
    #     if cls.batch_size == 1:
    #         padding_t = transforms.CenterPadTight(16)
    #     else:
    #         assert cls.eval_long_edge
    #         padding_t = transforms.CenterPad(cls.eval_long_edge)
    #
    #     orientation_t = None
    #     if cls.eval_orientation_invariant:
    #         orientation_t = transforms.DeterministicEqualChoice(
    #             [
    #                 None,
    #                 transforms.RotateBy90(fixed_angle=90),
    #                 transforms.RotateBy90(fixed_angle=180),
    #                 transforms.RotateBy90(fixed_angle=270),
    #             ],
    #             salt=3,
    #         )
    #
    #     return [
    #         transforms.NormalizeAnnotations(),
    #         rescale_t,
    #         padding_t,
    #         orientation_t,
    #     ]
    #
    # def _eval_preprocess(self):
    #     return transforms.Compose(
    #         [
    #             *self.common_eval_preprocess(),
    #             transforms.ToAnnotations(
    #                 [
    #                     transforms.ToKpAnnotations(
    #                         COCO_CATEGORIES,
    #                         keypoints_by_category={1: self.head_metas[0].keypoints},
    #                         skeleton_by_category={1: self.head_metas[1].skeleton},
    #                     ),
    #                     transforms.ToCrowdAnnotations(COCO_CATEGORIES),
    #                 ]
    #             ),
    #             transforms.EVAL_TRANSFORM,
    #         ]
    #     )
    #
    # def eval_loader(self):
    #     eval_data = Coco(
    #         image_dir=self.eval_image_dir,
    #         ann_file=self.eval_annotations,
    #         preprocess=self._eval_preprocess(),
    #         annotation_filter=self.eval_annotation_filter,
    #         min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
    #         category_ids=[1] if self.eval_annotation_filter else [],
    #     )
    #     return torch.utils.data.DataLoader(
    #         eval_data,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         pin_memory=self.pin_memory,
    #         num_workers=self.loader_workers,
    #         drop_last=False,
    #         collate_fn=collate_images_anns_meta,
    #     )

    # def metrics(self):
    #     return [
    #         metric.Coco(
    #             pycocotools.coco.COCO(self.eval_annotations),
    #             max_per_image=20,
    #             category_ids=[1],
    #             iou_type="keypoints",
    #         )
    #     ]
