import openpifpaf

from . import data_preprocessing
from . import datasets
from . import decoder
from . import encoder
from . import keypoint
from . import show
from . import transforms
from . import visualizer

from . import annotations
from . import headmeta
from . import match
from . import utils


# Overwrite pifpaf encoder configuration and add action encoder configuration
openpifapf_encoder_cli = openpifpaf.encoder.cli
openpifpaf_encoder_configure = openpifpaf.encoder.configure


def encoder_cli(parser):
    openpifapf_encoder_cli(parser)
    encoder.cli(parser)


def encoder_configure(args):
    openpifpaf_encoder_configure(args)
    encoder.configure(args)


openpifpaf.encoder.cli = encoder_cli
openpifpaf.encoder.configure = encoder_configure

# Create dummy painter for keypoint annotations as they otherwise conflict with the action annotation painter
class DummyPainter:
    def __init__(self, *args, **kwargs):
        pass

    def annotations(self, *args, **kwargs):
        pass


# The default pifpaf annotations inverse function does not work with action annotations
keypoint_annotations_inverse = openpifpaf.transforms.Preprocess.annotations_inverse


@staticmethod
def annotations_inverse(anns, meta):
    action_anns = [a for a in anns if isinstance(a, annotations.Action)]
    keypoint_anns = [a.keypoint_ann for a in action_anns]
    other_anns = [a for a in anns if not isinstance(a, annotations.Action)]

    keypoint_anns = keypoint_annotations_inverse(keypoint_anns, meta)
    other_anns = keypoint_annotations_inverse(other_anns, meta)

    for action_ann, kp_ann in zip(action_anns, keypoint_anns):
        action_ann.keypoint_ann = kp_ann

    other_anns.extend(action_anns)
    return other_anns


def register():
    openpifpaf.DATAMODULES["voc2012"] = datasets.voc2012.PascalVOC2012
    openpifpaf.DATAMODULES["stanford40"] = datasets.stanford40.Stanford40
    openpifpaf.network.HEADS[headmeta.Action] = openpifpaf.network.heads.CompositeField3
    openpifpaf.network.losses.LOSSES[
        headmeta.Action
    ] = openpifpaf.network.losses.CompositeLoss
    openpifpaf.decoder.DECODERS.add(decoder.Action)
    openpifpaf.show.annotation_painter.PAINTERS["Action"] = show.action.ActionPainter
    openpifpaf.show.annotation_painter.PAINTERS["Annotation"] = DummyPainter
    openpifpaf.transforms.Preprocess.annotations_inverse = annotations_inverse
