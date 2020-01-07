#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import sys
import copy
# import ftfy
# import gzip
import json
import math
import time
# import pickle
import string
import logging
import marshal
import sqlite3
import warnings
import numpy as np
import pandas as pd

from collections import (
    Counter,
    defaultdict,
    OrderedDict
)
from difflib import (
    SequenceMatcher
)

from itertools import (
    chain,
    combinations,
    zip_longest,
    product)

from operator import (
    lt, le,
    eq, ne,
    gt, ge,
    add, sub,
    mul, truediv
)

from io import (
    StringIO
)

from lxml import (
    etree
)

from subprocess import (
    check_output,
    STDOUT,
    CalledProcessError)

from types import (
    FunctionType
)

# -----------------------------------------------------------------------------
# helpers inside QP
# -----------------------------------------------------------------------------

from .core._constants import *

from .core.tools.functions import *

from .core.tools.logger import get_logger

# decorators
from .core.tools.qp_decorators import (
    lazy_property,
    params
)

# logics
from .core.logic import (
    has_any, has_all, has_count,
    not_any, not_all, not_count,
    is_lt, is_ne, is_gt,
    is_le, is_eq, is_ge,
    union, intersection, get_logic_index
)
