# encoding: utf-8

from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .dataset_loader import ImageDataset
from .msmt17 import MSMT17

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'msmt17': MSMT17,
    'dukemtmc': DukeMTMCreID
}


def get_names():
    return __factory.keys()


def init_dataset(name, path):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](root=path)
