import copy
import logging
import numpy as np

import openpifpaf
from openpifpaf.visualizer.base import Base
from openpifpaf.annotation import Annotation

from openpifpaf_action_prediction import headmeta
from openpifpaf_action_prediction import utils

try:
    import matplotlib.cm

    CMAP_ORANGES_NAN = copy.copy(matplotlib.cm.get_cmap("Oranges"))
    CMAP_ORANGES_NAN.set_bad("white", alpha=0.5)
except ImportError:
    CMAP_ORANGES_NAN = None

LOG = logging.getLogger(__name__)


class Aif(Base):
    show_margin = False
    show_confidences = False
    show_regressions = False
    show_background = False

    def __init__(self, meta: headmeta.Aif):
        super().__init__(meta.name)
        self.meta = meta
        keypoint_painter = openpifpaf.show.KeypointPainter(monocolor_connections=True)
        self.annotation_painter = openpifpaf.show.AnnotationPainter(
            painters={"Annotation": keypoint_painter}
        )

    def targets(self, field, *, annotation_dicts):
        # assert self.meta.keypoints is not None
        # assert self.meta.draw_skeleton is not None

        #

        confidences = field[:, 0]
        self._confidences(confidences, annotation_dicts)
        # self._regressions(field[:, 1:3], field[:, 4], annotations=annotations)

    # def predicted(self, field):
    #     self._confidences(field[:, 0])
    #     # self._regressions(
    #     #     field[:, 1:3],
    #     #     field[:, 4],
    #     #     annotations=self._ground_truth,
    #     #     confidence_fields=field[:, 0],
    #     #     uv_is_offset=False,
    #     # )

    def _confidences(self, confidences, annotations):
        if not self.show_confidences:
            return

        bboxes = [np.array(ann["bbox"]) for ann in annotations]

        for f in self.indices:
            # LOG.debug("%s", self.meta.keypoints[f])

            with self.image_canvas(
                self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]
            ) as ax:
                im = ax.imshow(
                    self.scale_scalar(confidences[f], self.meta.stride),
                    alpha=0.9,
                    vmin=0.0,
                    vmax=1.0,
                    cmap=CMAP_ORANGES_NAN,
                )
                self.colorbar(ax, im)

                for bbox in bboxes:
                    x, y = utils.bbox_center(bbox)
                    ax.scatter([x], [y], color="red")
                    plot_bbox(ax, bbox, color="cyan")

    # def _regressions(
    #     self,
    #     regression_fields,
    #     scale_fields,
    #     *,
    #     annotations=None,
    #     confidence_fields=None,
    #     uv_is_offset=True
    # ):
    #     if not self.show_regressions:
    #         return
    #
    #     for f in self.indices:
    #         LOG.debug("%s", self.meta.keypoints[f])
    #         confidence_field = (
    #             confidence_fields[f] if confidence_fields is not None else None
    #         )
    #
    #         with self.image_canvas(
    #             self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]
    #         ) as ax:
    #             openpifpaf.show.white_screen(ax, alpha=0.5)
    #             if annotations:
    #                 self.annotation_painter.annotations(
    #                     ax, annotations, color="lightgray"
    #                 )
    #             q = openpifpaf.show.quiver(
    #                 ax,
    #                 regression_fields[f, :2],
    #                 confidence_field=confidence_field,
    #                 xy_scale=self.meta.stride,
    #                 uv_is_offset=uv_is_offset,
    #                 cmap="Oranges",
    #                 clim=(0.5, 1.0),
    #                 width=0.001,
    #             )
    #             openpifpaf.show.boxes(
    #                 ax,
    #                 scale_fields[f] / 2.0,
    #                 confidence_field=confidence_field,
    #                 regression_field=regression_fields[f, :2],
    #                 xy_scale=self.meta.stride,
    #                 cmap="Oranges",
    #                 fill=False,
    #                 regression_field_is_offset=uv_is_offset,
    #             )
    #             if self.show_margin:
    #                 openpifpaf.show.margins(
    #                     ax, regression_fields[f, :6], xy_scale=self.meta.stride
    #                 )
    #
    #             self.colorbar(ax, q)


def plot_bbox(ax, bbox, **kwargs):
    x, y, w, h = bbox
    edges = [
        ([x, x], [y, y + h]),
        ([x, x + w], [y, y]),
        ([x, x + w], [y + h, y + h]),
        ([x + w, x + w], [y, y + h]),
    ]
    for xs, ys in edges:
        ax.plot(xs, ys, **kwargs)
