import dataclasses
import logging
from typing import ClassVar

import numpy as np
import torch

import openpifpaf
from openpifpaf.encoder.annrescaler import AnnRescaler
from openpifpaf.utils import mask_valid_area
from openpifpaf.visualizer import Cif as CifVisualizer

from openpifpaf_action_prediction import utils
from openpifpaf_action_prediction import headmeta
from openpifpaf_action_prediction import visualizer

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class AifCenter:
    meta: headmeta.AifCenter
    rescaler: AnnRescaler = None
    v_threshold: int = 0
    bmin: float = 1.0  #: in pixels
    visualizer: CifVisualizer = None

    side_length: ClassVar[float] = 0.1
    padding: ClassVar[int] = 10

    def __call__(self, image, anns, meta):
        return AifCenterGenerator(self)(image, anns, meta)


class AifCenterGenerator:

    mask_background = True
    mask_unannotated = True

    def __init__(self, config: AifCenter):
        self.config = config
        self.rescaler = config.rescaler or AnnRescaler(
            config.meta.stride, config.meta.pose
        )
        self.visualizer = config.visualizer or visualizer.aif.Aif(config.meta)
        self.visualizer.show_confidences = True
        self.intensities = None

    def __call__(self, image, anns, meta):
        shape = image.shape[2:0:-1]
        width = shape[0]
        height = shape[1]

        self.init_fields(width, height)
        self.fill_fields(anns)
        self.mask_regions(anns)

        # TODO: does this the right thing
        valid_area = self.rescaler.valid_area(meta)
        fields = self.fields(valid_area)

        self.visualizer.processed_image(image)
        self.visualizer.targets(fields, annotation_dicts=anns)

        return fields

    def init_fields(self, width, height):
        n_fields = self.config.meta.n_fields
        stride = self.config.meta.stride
        padding = self.config.padding

        width = (width - 1) // stride + 1
        height = (height - 1) // stride + 1
        field_width = width + 2 * padding
        field_height = height + 2 * padding

        self.intensities = np.zeros(
            (n_fields, field_height, field_width), dtype=np.float32
        )

    def fill_fields(self, anns):
        for ann in anns:
            if "actions" not in ann:
                continue

            keypoint_indices = self.config.meta.keypoint_indices
            centers = np.array(
                utils.keypoint_centers(ann["keypoints"], keypoint_indices)
            )
            centers = centers / self.config.meta.stride

            action_labels = self.config.meta.action_labels(ann["actions"])
            action_mask = np.asarray(action_labels).astype(bool)

            area = utils.bbox_area(ann["bbox"])
            scale = np.sqrt(area) / self.config.meta.stride

            for center in centers:
                self.fill_center(center, action_mask, scale)

    def fill_center(self, center, action_mask, scale):

        radius = int(np.round(max(0, scale * self.config.side_length)))
        side_length = 2 * radius + 1

        i, j = np.round(center - radius).astype(np.int) + self.config.padding
        minj, mini, widthj, widthi = utils.clip_box(
            self.intensities, [j, i, side_length, side_length]
        )
        self.intensities[action_mask, minj : minj + widthj, mini : mini + widthi] = 1.0

    def mask_regions(self, anns):
        mask = np.zeros(self.intensities[0].shape).astype(bool)
        no_action = [a for a in anns if "actions" not in a]
        action = [a for a in anns if "actions" in a]

        # hide background
        if self.mask_background:
            mask[...] = True

        # unhide annotated regions
        for ann in action:
            x, y, w, h = ann["bbox"]
            x_min, y_min, x_max, y_max = (
                np.array([x, y, x + w, y + h]) / self.config.meta.stride
            )
            bbox = np.array(
                [
                    np.floor(x_min),
                    np.floor(y_min),
                    np.ceil(x_max + 1),
                    np.ceil(y_max + 1),
                ]
            ).astype(int)
            i_min, j_min, i_max, j_max = bbox + self.config.padding
            mask[j_min:j_max, i_min:i_max] = False

        # hide unannotated regions
        if self.mask_unannotated:
            for ann in no_action:
                x, y, w, h = ann["bbox"]
                x_min, y_min, x_max, y_max = (
                    np.array([x, y, x + w, y + h]) / self.config.meta.stride
                )
                bbox = np.array(
                    [
                        np.floor(x_min),
                        np.floor(y_min),
                        np.ceil(x_max + 1),
                        np.ceil(y_max + 1),
                    ]
                ).astype(int)
                i_min, j_min, i_max, j_max = bbox + self.config.padding
                mask[j_min:j_max, i_min:i_max] = True

        self.intensities[:, mask] = np.nan

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        mask_valid_area(intensities, valid_area, fill_value=np.nan)
        return torch.from_numpy(np.expand_dims(intensities, 1))


@dataclasses.dataclass
class AifCif(openpifpaf.encoder.Cif):
    def __call__(self, image, anns, meta):
        anns = [a for a in anns if "keypoints" in a]
        return openpifpaf.encoder.Cif.__call__(self, image, anns, meta)


@dataclasses.dataclass
class AifCaf(openpifpaf.encoder.Caf):
    def __call__(self, image, anns, meta):
        anns = [a for a in anns if "keypoints" in a]
        return openpifpaf.encoder.Caf.__call__(self, image, anns, meta)


def cli(parser):
    group = parser.add_argument_group("AifCenter Encoder")
    group.add_argument("--aif-encoder-side-length", default=AifCenter.side_length)
    group.add_argument(
        "--aif-encoder-no-mask-background",
        default=not AifCenterGenerator.mask_background,
        action="store_true",
    )
    group.add_argument(
        "--aif-encoder-no-mask-unannotated",
        default=not AifCenterGenerator.mask_unannotated,
        action="store_true",
    )


def configure(args):
    AifCenter.side_length = args.aif_encoder_side_length
    AifCenterGenerator.mask_background = not args.aif_encoder_no_mask_background
    AifCenterGenerator.mask_unannotated = not args.aif_encoder_no_mask_unannotated
