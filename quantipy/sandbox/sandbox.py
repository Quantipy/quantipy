#  -*- coding: utf-8 -*-

from quantipy.core.chain import Chain as ChainCore
from quantipy.core.chainmanager import ChainManager as ChainManagerCore
import warnings
warnings.simplefilter('always')


class ChainManager(ChainManagerCore):

    def __init__(self, stack):
        msg = "Please use 'quantipy.core.chainmanager.ChainManager' instead!"
        warnings.warn(msg, DeprecationWarning)
        super(ChainManager, self).__init__(stack)


class Chain(ChainCore):

    def __init__(self, stack, name, structure=None):
        msg = "Please use 'quantipy.core.chain.Chain' instead!"
        warnings.warn(msg, DeprecationWarning)
        super(Chain, self).__init__(stack, name, structure=None)
