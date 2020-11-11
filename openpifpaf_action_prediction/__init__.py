import openpifpaf
import importlib

from . import headmeta
from . import utils
from . import datasets

# openpifpaf.network.factory module is overwritten in openpifpaf.network.__init__
# need to import it manually
network_factory = importlib.import_module("openpifpaf.network.factory")


def register():
    openpifpaf.DATAMODULES["vcoco"] = datasets.vcoco.Vcoco
    network_factory.HEAD_FACTORIES[
        headmeta.Aif
    ] = openpifpaf.network.heads.CompositeField3
