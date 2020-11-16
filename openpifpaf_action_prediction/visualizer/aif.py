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

    def __init__(self, meta: headmeta.AifCenter):
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
