#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
savReaderWriter: A cross-platform Python interface to the IBM SPSS
Statistics Input Output Module. Read or Write SPSS system files (.sav, .zsav)

.. moduleauthor:: Albert-Jan Roskam <fomcl "at" yahoo "dot" com>

"""

# change this to 'True' in case you experience segmentation
# faults related to freeing memory.
segfaults = False

import os
import sys

try:
    import psyco
    psycoOk = True  # reading 66 % faster
except ImportError:
    psycoOk = False
try:
    import numpy
    numpyOk = True
except ImportError:
    numpyOk = False

try:
    from cWriterow import cWriterow  # writing 66 % faster
    cWriterowOK = True
except ImportError:
    cWriterowOK = False

__author__ = "Albert-Jan Roskam" + " " + "@".join(["fomcl", "yahoo.com"])
__version__ = open(os.path.join(os.path.dirname(__file__),
                                "VERSION")).read().strip()


allFormats = {
    1: (b"SPSS_FMT_A", b"Alphanumeric"),
    2: (b"SPSS_FMT_AHEX", b"Alphanumeric hexadecimal"),
    3: (b"SPSS_FMT_COMMA", b"F Format with commas"),
    4: (b"SPSS_FMT_DOLLAR", b"Commas and floating dollar sign"),
    5: (b"SPSS_FMT_F", b"Default Numeric Format"),
    6: (b"SPSS_FMT_IB", b"Integer binary"),
    7: (b"SPSS_FMT_PIBHEX", b"Positive integer binary - hex"),
    8: (b"SPSS_FMT_P", b"Packed decimal"),
    9: (b"SPSS_FMT_PIB", b"Positive integer binary unsigned"),
    10: (b"SPSS_FMT_PK", b"Positive integer binary unsigned"),
    11: (b"SPSS_FMT_RB", b"Floating point binary"),
    12: (b"SPSS_FMT_RBHEX", b"Floating point binary hex"),
    15: (b"SPSS_FMT_Z", b"Zoned decimal"),
    16: (b"SPSS_FMT_N", b"N Format- unsigned with leading 0s"),
    17: (b"SPSS_FMT_E", b"E Format- with explicit power of 10"),
    20: (b"SPSS_FMT_DATE", b"Date format dd-mmm-yyyy"),
    21: (b"SPSS_FMT_TIME", b"Time format hh:mm:ss.s"),
    22: (b"SPSS_FMT_DATETIME", b"Date and Time"),
    23: (b"SPSS_FMT_ADATE", b"Date format dd-mmm-yyyy"),
    24: (b"SPSS_FMT_JDATE", b"Julian date - yyyyddd"),
    25: (b"SPSS_FMT_DTIME", b"Date-time dd hh:mm:ss.s"),
    26: (b"SPSS_FMT_WKDAY", b"Day of the week"),
    27: (b"SPSS_FMT_MONTH", b"Month"),
    28: (b"SPSS_FMT_MOYR", b"mmm yyyy"),
    29: (b"SPSS_FMT_QYR", b"q Q yyyy"),
    30: (b"SPSS_FMT_WKYR", b"ww WK yyyy"),
    31: (b"SPSS_FMT_PCT", b"Percent - F followed by %"),
    32: (b"SPSS_FMT_DOT", b"Like COMMA, switching dot for comma"),
    33: (b"SPSS_FMT_CCA", b"User Programmable currency format"),
    34: (b"SPSS_FMT_CCB", b"User Programmable currency format"),
    35: (b"SPSS_FMT_CCC", b"User Programmable currency format"),
    36: (b"SPSS_FMT_CCD", b"User Programmable currency format"),
    37: (b"SPSS_FMT_CCE", b"User Programmable currency format"),
    38: (b"SPSS_FMT_EDATE", b"Date in dd/mm/yyyy style"),
    39: (b"SPSS_FMT_SDATE", b"Date in yyyy/mm/dd style")}

MAXLENGTHS = {
    "SPSS_MAX_VARNAME": (64, "Variable name"),
    "SPSS_MAX_SHORTVARNAME": (8, "Short (compatibility) variable name"),
    "SPSS_MAX_SHORTSTRING": (8, "Short string variable"),
    "SPSS_MAX_IDSTRING": (64, "File label string"),
    "SPSS_MAX_LONGSTRING": (32767, "Long string variable"),
    "SPSS_MAX_VALLABEL": (120, "Value label"),
    "SPSS_MAX_VARLABEL": (256, "Variable label"),
    "SPSS_MAX_7SUBTYPE": (40, "Maximum record 7 subtype"),
    "SPSS_MAX_ENCODING": (64, "Maximum encoding text")}

supportedDates = {  # uses ISO dates wherever applicable.
    b"DATE": "%Y-%m-%d",
    b"JDATE": "%Y-%m-%d",
    b"EDATE": "%Y-%m-%d",
    b"SDATE": "%Y-%m-%d",
    b"DATETIME": "%Y-%m-%d %H:%M:%S",
    b"ADATE": "%Y-%m-%d",
    b"WKDAY": "%A",
    b"MONTH": "%B",
    b"MOYR": "%B %Y",
    b"WKYR": "%W WK %Y",
    b"QYR": "%m Q %Y",  # %m (month) is converted to quarter, see next dict.
    b"TIME": "%H:%M:%S.%f",
    b"DTIME": "%d %H:%M:%S"}

QUARTERS = {b'01': b'1', b'02': b'1', b'03': b'1', 
            b'04': b'2', b'05': b'2', b'06': b'2',
            b'07': b'3', b'08': b'3', b'09': b'3', 
            b'10': b'4', b'11': b'4', b'12': b'4'}

userMissingValues = {
    "SPSS_NO_MISSVAL": 0,
    "SPSS_ONE_MISSVAL": 1,
    "SPSS_TWO_MISSVAL": 2,
    "SPSS_THREE_MISSVAL": 3,
    "SPSS_MISS_RANGE": -2,
    "SPSS_MISS_RANGEANDVAL": -3}

version = __version__

sys.path.insert(0, os.path.dirname(__file__))
from py3k import *
from error import *
from generic import *
from header import *
from savReader import *
from savWriter import *
from savHeaderReader import *

__all__ = ["SavReader", "SavWriter", "SavHeaderReader"]
