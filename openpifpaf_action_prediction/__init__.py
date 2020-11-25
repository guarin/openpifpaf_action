import openpifpaf
import importlib

from . import datasets
from . import decoder
from . import encoder
from . import metrics
from . import transforms
from . import visualizer

from . import annotations
from . import headmeta
from . import utils

# openpifpaf.network.factory module is overwritten in openpifpaf.network.__init__
# need to import it manually
network_factory = importlib.import_module("openpifpaf.network.factory")
decoder_factory = importlib.import_module("openpifpaf.decoder.factory")


def register():
    openpifpaf.DATAMODULES["vcoco"] = datasets.vcoco.Vcoco
    openpifpaf.DATAMODULES["voc2012"] = datasets.pascal_voc_2012.PascalVOC2012
    network_factory.HEAD_FACTORIES[
        headmeta.AifCenter
    ] = openpifpaf.network.heads.CompositeField3
    decoder_factory.DECODERS.add(decoder.aif.AifCenter)

    openpifpaf.transforms.Preprocess.annotations_inverse = (
        lambda annotations, meta: annotations
    )
