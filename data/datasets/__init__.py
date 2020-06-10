# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .veri import VeRi
from .wicrep import WicrepDataset
from .dataset_loader import ImageDataset
from .composite import CompositeDataset

__factory = {
    'market1501': Market1501,
    # 'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': VeRi,
    'wicrep': WicrepDataset,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)


def init_composite_dataset(names, *args, **kwargs):
    datasets = []
    for name in names:
        if name not in __factory.keys():
            raise KeyError("Unknown datasets: {}".format(name))
        datasets.append(__factory[name](*args, **kwargs))
    return CompositeDataset(datasets)
