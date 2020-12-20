import openpifpaf
import importlib

from . import datasets
from . import decoder
from . import encoder
from . import metrics
from . import show
from . import transforms
from . import visualizer

from . import annotations
from . import headmeta
from . import match
from . import utils

keypoint_annotations_inverse = openpifpaf.transforms.Preprocess.annotations_inverse


# overwrite pifpaf encoder configuration
openpifapf_encoder_cli = openpifpaf.encoder.cli
openpifpaf_encoder_configure = openpifpaf.encoder.configure


def encoder_cli(parser):
    openpifapf_encoder_cli(parser)
    encoder.aif.cli(parser)


def encoder_configure(args):
    openpifpaf_encoder_configure(args)
    encoder.aif.configure(args)


openpifpaf.encoder.cli = encoder_cli
openpifpaf.encoder.configure = encoder_configure


class DummyPainter:
    def __init__(self, *args, **kwargs):
        pass

    def annotations(self, *args, **kwargs):
        pass


@staticmethod
def annotations_inverse(anns, meta):
    aif_anns = [a for a in anns if isinstance(a, annotations.AifCenter)]
    keypoint_anns = [a.keypoint_ann for a in aif_anns]
    other_anns = [a for a in anns if not isinstance(a, annotations.AifCenter)]

    keypoint_anns = keypoint_annotations_inverse(keypoint_anns, meta)
    other_anns = keypoint_annotations_inverse(other_anns, meta)

    for aif_ann, kp_ann in zip(aif_anns, keypoint_anns):
        aif_ann.keypoint_ann = kp_ann

    other_anns.extend(aif_anns)
    return other_anns


def register():
    openpifpaf.DATAMODULES["voc2012"] = datasets.voc2012.PascalVOC2012
    openpifpaf.DATAMODULES["stanford40"] = datasets.stanford40.Stanford40
    openpifpaf.network.HEAD_FACTORIES[
        headmeta.AifCenter
    ] = openpifpaf.network.heads.DeepCompositeField3
    openpifpaf.decoder.DECODERS.add(decoder.aif.AifCenter)
    openpifpaf.show.annotation_painter.PAINTERS["AifCenter"] = show.aif.AifPainter
    openpifpaf.show.annotation_painter.PAINTERS["Annotation"] = DummyPainter
    # annotations_inverse = openpifpaf.transforms.Preprocess.annotations_inverse
    # openpifpaf.show.annotation_painter.KeypointPainter.annotations = (
    #     lambda *args, **kwargs: None
    # )
    openpifpaf.transforms.Preprocess.annotations_inverse = annotations_inverse
