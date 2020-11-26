import dataclasses
import logging
from typing import ClassVar

import numpy as np
import torch

import openpifpaf
from openpifpaf.encoder.annrescaler import AnnRescaler
from openpifpaf.utils import create_sink, mask_valid_area
from openpifpaf.visualizer import Cif as CifVisualizer

from openpifpaf_action_prediction import headmeta
from openpifpaf_action_prediction import datasets
from openpifpaf_action_prediction import visualizer

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class AifCenter:
    meta: headmeta.AifCenter
    rescaler: AnnRescaler = None
    v_threshold: int = 0
    bmin: float = 1.0  #: in pixels
    visualizer: CifVisualizer = None

    side_length: ClassVar[int] = 10
    padding: ClassVar[int] = 10

    def __call__(self, image, anns, meta):
        return AifCenterGenerator(self)(image, anns, meta)


class AifCenterGenerator:
    def __init__(self, config: AifCenter):
        self.config = config

        self.rescaler = config.rescaler or AnnRescaler(
            config.meta.stride, config.meta.pose
        )
        self.visualizer = config.visualizer or visualizer.aif.Aif(config.meta)
        self.visualizer.show_confidences = True

        # TODO: check person keypoint rescaling
        self.intensities = None

        self.sink = create_sink(config.side_length)
        self.s_offset = (config.side_length - 1.0) / 2.0

    def __call__(self, image, anns, meta):
        shape = image.shape[2:0:-1]
        width = shape[0]
        height = shape[1]

        self.init_fields(width, height)
        self.fill_fields(anns)
        self.mask_unnanotated(anns)

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
            if "action_labels" not in ann:
                continue

            keypoint_indices = self.config.meta.keypoint_indices
            keypoints = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
            scale = self.rescaler.scale(keypoints) / self.config.meta.stride
            keypoints = keypoints[keypoint_indices, :2]
            center = keypoints.mean(0)

            center = center / self.config.meta.stride

            action_labels = datasets.utils.filter_action_labels(
                ann["action_labels"],
                self.config.meta.actions,
                self.config.meta.action_dict,
            )

            action_mask = np.asarray(action_labels).astype(bool)
            scale = max(1.0, scale / 5)
            self.fill_center(center, action_mask, scale)

    def mask_unnanotated(self, anns):
        mask = np.zeros(self.intensities[0].shape).astype(bool)
        no_action = [a for a in anns if "action_labels" not in a]
        action = [a for a in anns if "action_labels" in a]

        # hide unannotated regions
        for ann in no_action:
            x, y, w, h = ann["bbox"]
            x_min, y_min, x_max, y_max = (
                np.array([x, y, x + w, y + h]) / self.config.meta.stride
            )
            bbox = np.array(
                [np.floor(x_min), np.floor(y_min), np.ceil(x_max), np.ceil(y_max)]
            ).astype(int)
            i_min, j_min, i_max, j_max = bbox + self.config.padding
            mask[j_min:j_max, i_min:i_max] = True

        # unhide annotated regions
        for ann in action:
            x, y, w, h = ann["bbox"]
            x_min, y_min, x_max, y_max = (
                np.array([x, y, x + w, y + h]) / self.config.meta.stride
            )
            bbox = np.array(
                [np.floor(x_min), np.floor(y_min), np.ceil(x_max), np.ceil(y_max)]
            ).astype(int)
            i_min, j_min, i_max, j_max = bbox + self.config.padding
            mask[j_min:j_max, i_min:i_max] = False
        self.intensities[:, mask] = np.nan

    def fill_center(self, center, action_mask, scale):
        ij = np.round(center - self.s_offset).astype(np.int) + self.config.padding

        side_length = self.config.side_length
        if (
            (ij.min() < 0)
            or (ij[0] + side_length > self.intensities.shape[2])
            or (ij[1] + side_length > self.intensities.shape[1])
        ):
            return

        offset = center - (ij + self.s_offset - self.config.padding)
        offset = offset.reshape(-1, 2, 1, 1)

        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=1)
        mask = sink_l < (0.71 * scale)

        i, j = ij[0], ij[1]
        self.intensities[action_mask, j : j + side_length, i : i + side_length] += mask
        # convert intensities back to 0 or 1
        self.intensities = (self.intensities > 0).astype(np.float32)

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        mask_valid_area(intensities, valid_area)
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
