#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings

from savReaderWriter import *

retcodes = {
    0: "SPSS_OK",
    1: "SPSS_FILE_OERROR",
    2: "SPSS_FILE_WERROR",
    3: "SPSS_FILE_RERROR",
    4: "SPSS_FITAB_FULL",
    5: "SPSS_INVALID_HANDLE",
    6: "SPSS_INVALID_FILE",
    7: "SPSS_NO_MEMORY",
    8: "SPSS_OPEN_RDMODE",
    9: "SPSS_OPEN_WRMODE",
    10: "SPSS_INVALID_VARNAME",
    11: "SPSS_DICT_EMPTY",
    12: "SPSS_VAR_NOTFOUND",
    13: "SPSS_DUP_VAR",
    14: "SPSS_NUME_EXP",
    15: "SPSS_STR_EXP",
    16: "SPSS_SHORTSTR_EXP",
    17: "SPSS_INVALID_VARTYPE",
    18: "SPSS_INVALID_MISSFOR",
    19: "SPSS_INVALID_COMPSW",
    20: "SPSS_INVALID_PRFOR",
    21: "SPSS_INVALID_WRFOR",
    22: "SPSS_INVALID_DATE",
    23: "SPSS_INVALID_TIME",
    24: "SPSS_NO_VARIABLES",
    27: "SPSS_DUP_VALUE",
    28: "SPSS_INVALID_CASEWGT",
    30: "SPSS_DICT_COMMIT",
    31: "SPSS_DICT_NOTCOMMIT",
    33: "SPSS_NO_TYPE2",
    41: "SPSS_NO_TYPE73",
    45: "SPSS_INVALID_DATEINFO",
    46: "SPSS_NO_TYPE999",
    47: "SPSS_EXC_STRVALUE",
    48: "SPSS_CANNOT_FREE",
    49: "SPSS_BUFFER_SHORT",
    50: "SPSS_INVALID_CASE",
    51: "SPSS_INTERNAL_VLABS",
    52: "SPSS_INCOMPAT_APPEND",
    53: "SPSS_INTERNAL_D_A",
    54: "SPSS_FILE_BADTEMP",
    55: "SPSS_DEW_NOFIRST",
    56: "SPSS_INVALID_MEASURELEVEL",
    57: "SPSS_INVALID_7SUBTYPE",
    58: "SPSS_INVALID_VARHANDLE",
    59: "SPSS_INVALID_ENCODING",
    60: "SPSS_FILES_OPEN",
    70: "SPSS_INVALID_MRSETDEF",
    71: "SPSS_INVALID_MRSETNAME",
    72: "SPSS_DUP_MRSETNAME",
    73: "SPSS_BAD_EXTENSION",
    74: "SPSS_INVALID_EXTENDEDSTRING",
    75: "SPSS_INVALID_ATTRNAME",
    76: "SPSS_INVALID_ATTRDEF",
    77: "SPSS_INVALID_MRSETINDEX",
    78: "SPSS_INVALID_VARSETDEF",
    79: "SPSS_INVALID_ROLE",

    -15: "SPSS_EMPTY_DEW",
    -14: "SPSS_NO_DEW",
    -13: "SPSS_EMPTY_MULTRESP",
    -12: "SPSS_NO_MULTRESP",
    -11: "SPSS_NO_DATEINFO",
    -10: "SPSS_NO_CASEWGT",
    -9: "SPSS_NO_LABEL",
    -8: "SPSS_NO_LABELS",
    -7: "SPSS_EMPTY_VARSETS",
    -6: "SPSS_NO_VARSETS",
    -5: "SPSS_FILE_END",
    -4: "SPSS_EXC_VALLABEL",
    -3: "SPSS_EXC_LEN120",
    -2: "SPSS_EXC_VARLABEL",
    -1: "SPSS_EXC_LEN64"}

class SPSSIOError(Exception):
    """
    Error class for the IBM SPSS Statistics Input Output Module
    """
    def __init__(self, msg="Unknown", retcode=None):
        self.retcode = retcode
        Exception.__init__(self, msg)

class SPSSIOWarning(UserWarning):
    """
    Warning class for the IBM SPSS Statistics Input Output Module

    If the environment variable SAVRW_DISPLAY_WARNS is undefined or 0
    (False) warnings are ignored. If it is 1 (True) warnings (non-fatal
    exceptions) are displayed once for every location where they occur.
    """
    pass

# Warnings are usually harmless!
action = os.environ.get("SAVRW_DISPLAY_WARNS")
action = action.lower() if action else "ignore"
actions = ("error", "ignore", "always", "default", "module", "once")
if action not in actions:
   raise ValueError("If set, environment variable SAVRW_DISPLAY_WARNS must be"
                    " one of the following values:\n" % ", ".join(actions))
warnings.simplefilter(action, SPSSIOWarning)

def checkErrsWarns(msg, retcode):
    """Throws a warning if retcode < 0 (and warnings are not ignored),
    and throws an error if retcode > 0. Returns None if retcode == 0"""
    if not retcode:
        return
    msg += " [%s]" % retcodes.get(retcode, retcode)
    if retcode > 0:
        raise SPSSIOError(msg, retcode)
    elif retcode < 0:
        warnings.warn(msg, SPSSIOWarning, stacklevel=2)
