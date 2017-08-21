#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ctypes import *
import ctypes.util
import struct
import sys
import os
import re
import time
import getpass
import functools
import gc

from savReaderWriter import *
from generic import *

__version__ = version

class Header(Generic):

    """
    This class contains methods responsible for getting and setting meta data
    that is embedded in the IBM SPSS Statistics data file. In SPSS speak, this
    header information is known as the SPSS Data Dictionary (which has diddly
    squat to do with a Python dictionary!).
    """

    def __init__(self, savFileName, mode, refSavFileName, ioUtf8, ioLocale=None):
        """Constructor"""
        super(Header, self).__init__(savFileName, ioUtf8, ioLocale)
        self.spssio = self.loadLibrary()
        self.libc = cdll.LoadLibrary(ctypes.util.find_library("c"))
        self.fh = super(Header, self).openSavFile(savFileName, mode,
                                                  refSavFileName)
        self.varNames, self.varTypes = self.varNamesTypes
        self.vNames = dict(zip(self.varNames, self.encode(self.varNames)))

    def openSavFile(self):
        """This function returns the file handle that was opened in the
        super class"""
        return self.fh

    def decode(func):
        """Decorator to Utf-8 decode all str items contained in a dictionary
        If ioUtf8=True, the dictionary's keys and values are decoded, but only
        values that are strs, lists, or dicts."""
        bytes_ = bytes if sys.version_info[0] > 2 else str
        uS = lambda x: x.decode("utf-8") if isinstance(x, bytes_) else x
        uL = lambda x: map(uS, x) if isinstance(x, list) else x
        @functools.wraps(func)
        def wrapper(arg):
            result = func(arg)
            if not arg.ioUtf8:
                return result  # unchanged
            if isinstance(result, bytes_):
                return uS(result)
            uresult = {}
            for k, v in result.items():
                uresult[uS(k)] = {}
                try:
                    for i, j in v.items():  # or wrapper(j) recursion?
                        uresult[uS(k)][uS(i)] = uS(uL(j))
                except AttributeError:
                    uresult[uS(k)] = uL(uS(v))
            return uresult
        return wrapper

    def encode(self, item):
        """Counter part of decode helper function, does the opposite of that
        function (but is not a decorator)"""
        if not self.ioUtf8:
            return item  # unchanged
        u = str if isPy3k else unicode
        utf8dify = lambda x: x.encode("utf-8") if isinstance(x, u) else x
        if isinstance(item, list):
            return list(map(utf8dify, item))
        elif isinstance(item, dict):
            return dict([(utf8dify(x), utf8dify(y)) for x, y in item.items()])
        return utf8dify(item)

    def freeMemory(self, funcName, *args):
        """Clean up: free memory claimed by e.g. getValueLabels and
        getVarNamesTypes"""
        gc.collect()
        if segfaults:
            return
        #print("... freeing", funcName[8:])
        func = getattr(self.spssio, funcName)
        retcode = func(*args)
        if retcode:
            checkErrsWarns("Problem freeing memory using %s" % funcName, retcode)

    @property
    def numberofCases(self):
        """This function reports the number of cases present in a data file.
        Prehistoric files (< SPSS v6.0) don't contain nCases info, therefore
        a guesstimate of the number of cases is given for those files"""
        nCases = c_long()
        func = self.spssio.spssGetNumberofCases
        retcode = func(c_int(self.fh), byref(nCases))
        if nCases.value == -1:
            func = self.spssio.spssGetEstimatedNofCases
            retcode = func(c_int(self.fh), byref(nCases))
        if retcode:
            checkErrsWarns("Problem getting number of cases", retcode)
        return nCases.value

    @property
    def numberofVariables(self):
        """This function returns the number of variables (columns) in the
        spss dataset"""
        numVars = c_int()
        func = self.spssio.spssGetNumberofVariables
        retcode = func(c_int(self.fh), byref(numVars))
        if retcode:
            checkErrsWarns("Problem getting number of variables", retcode)
        return numVars.value

    @property
    def varNamesTypes(self):
        """Get/Set variable names and types.
        --Variable names is a list of the form ['var1', var2', 'etc']
        --Variable types is a dictionary of the form {varName: varType}
        The variable type code is an integer in the range 0-32767, 0
        indicating a numeric variable (F8.2) and a positive value
        indicating a string variable of that size (in bytes)."""
        if hasattr(self, "varNames"):
            return self.varNames, self.varTypes

        # initialize arrays
        numVars = self.numberofVariables
        numVars_ = c_int(numVars)
        varNamesArr = (POINTER(c_char_p * numVars))()
        varTypesArr = (POINTER(c_int * numVars))()

        # get variable names
        func = self.spssio.spssGetVarNames
        retcode = func(c_int(self.fh), byref(numVars_),
                       byref(varNamesArr), byref(varTypesArr))
        if retcode:
            checkErrsWarns("Problem getting variable names & types", retcode)

        # get array contents
        varNames = [varNamesArr[0][i] for i in xrange(numVars)]
        varTypes = [varTypesArr[0][i] for i in xrange(numVars)]
        if self.ioUtf8:
            varNames = [varName.decode("utf-8") for varName in varNames]

        # clean up
        args = (varNamesArr, varTypesArr, numVars)
        self.freeMemory("spssFreeVarNames", *args)

        return varNames, dict(zip(varNames, varTypes))

    @varNamesTypes.setter
    def varNamesTypes(self, varNamesVarTypes):
        badLengthMsg = ("Empty or longer than %s chars" %
                        (MAXLENGTHS['SPSS_MAX_VARNAME'][0]))
        varNames, varTypes = varNamesVarTypes
        varNameRetcodes = {
            0: ('SPSS_NAME_OK', 'Valid standard name'),
            1: ('SPSS_NAME_SCRATCH', 'Valid scratch var name'),
            2: ('SPSS_NAME_SYSTEM', 'Valid system var name'),
            3: ('SPSS_NAME_BADLTH', badLengthMsg),
            4: ('SPSS_NAME_BADCHAR', 'Invalid character or embedded blank'),
            5: ('SPSS_NAME_RESERVED', 'Name is a reserved word'),
            6: ('SPSS_NAME_BADFIRST', 'Invalid initial char (otherwise OK)')}
        validate = self.spssio.spssValidateVarname
        func = self.spssio.spssSetVarName
        for varName in self.varNames:
            varLength = self.varTypes[varName]
            retcode = validate(c_char_py3k(varName))
            if retcode:
                msg = ("%r is an invalid variable name [%r]" %
                       (varName, ": ".join(varNameRetcodes.get(retcode))))
                raise SPSSIOError(msg, retcode)
            retcode = func(c_int(self.fh), c_char_py3k(varName), c_int(varLength))
            if retcode:
                msg = "Problem setting variable name %r" % varName
                checkErrsWarns(msg, retcode)

    @property
    @decode
    def valueLabels(self):
        """Get/Set VALUE LABELS.
        Takes a dictionary of the form {varName: {value: valueLabel}:
        --{'numGender': {1: 'female', {2: 'male'}}
        --{'strGender': {'f': 'female', 'm': 'male'}}"""
        def initArrays(isNumeric=True, size=1000):
            """default size assumes never more than 1000 labels"""
            labelsArr = (POINTER(c_char_p * size))()
            if isNumeric:
                return (POINTER(c_double * size))(), labelsArr
            return (POINTER(c_char_p * size))(), labelsArr

        valueLabels = {}
        for varName in self.varNames:
            vName = self.vNames[varName]
            numLabels = c_int()

            # step 1: get array size (numeric values)
            if self.varTypes[varName] == 0:
                valuesArr, labelsArr = initArrays(True)
                func = self.spssio.spssGetVarNValueLabels
                retcode = func(c_int(self.fh), c_char_py3k(vName),
                               byref(valuesArr), byref(labelsArr),
                               byref(numLabels))
                valuesArr, labelsArr = initArrays(True, numLabels.value)

            # step 1: get array size (string values)
            else:
                valuesArr, labelsArr = initArrays(False)
                func = self.spssio.spssGetVarCValueLabels
                retcode = func(c_int(self.fh), c_char_py3k(vName),
                               byref(valuesArr), byref(labelsArr),
                               byref(numLabels))
                valuesArr, labelsArr = initArrays(False, numLabels.value)

            # step 2: get labels with array of proper size
            retcode = func(c_int(self.fh), c_char_py3k(vName), byref(valuesArr),
                           byref(labelsArr), byref(numLabels))
            if retcode:
                msg = "Problem getting value labels of variable %r"  % varName
                checkErrsWarns(msg, retcode)

            # get array contents
            if not numLabels.value:
                continue
            values = [valuesArr[0][i] for i in xrange(numLabels.value)]
            labels = [labelsArr[0][i] for i in xrange(numLabels.value)]
            valueLabelsX = [(val, lbl) for val, lbl in zip(values, labels)]
            valueLabels[varName] = dict(valueLabelsX)

            # clean up
            args = (valuesArr, labelsArr, numLabels)
            if self.varTypes[varName] == 0:
                self.freeMemory("spssFreeVarNValueLabels", *args)
            else:
                self.freeMemory("spssFreeVarCValueLabels", *args)

        return valueLabels

    @valueLabels.setter
    def valueLabels(self, valueLabels):
        if not valueLabels:
            return
        valLabN = self.spssio.spssSetVarNValueLabel
        valLabC = self.spssio.spssSetVarCValueLabel
        valueLabels = self.encode(valueLabels)
        for varName, valueLabelsX in valueLabels.items():
            valueLabelsX = self.encode(valueLabelsX)
            for value, label in valueLabelsX.items():
                if self.varTypes[varName] == 0:
                    retcode = valLabN(c_int(self.fh), c_char_py3k(varName),
                                      c_double(value), c_char_py3k(label))
                else:
                    retcode = valLabC(c_int(self.fh), c_char_py3k(varName),
                                      c_char_py3k(value), c_char_py3k(label))
                if retcode:
                    msg = "Problem setting value labels of variable %r"
                    checkErrsWarns(msg % varName, retcode)

    @property
    @decode
    def varLabels(self):
        """Get/set VARIABLE LABELS.
        Returns/takes a dictionary of the form {varName: varLabel}.
        For example: varLabels = {'salary': 'Salary (dollars)',
                                  'educ': 'Educational level (years)'}"""
        lenBuff = MAXLENGTHS['SPSS_MAX_VARLABEL'][0]
        varLabel = create_string_buffer(lenBuff)
        func = self.spssio.spssGetVarLabelLong
        varLabels = {}
        for varName in self.varNames:
            vName = self.vNames[varName]
            retcode = func(c_int(self.fh), c_char_py3k(vName),
                           byref(varLabel), c_int(lenBuff), byref(c_int()))
            varLabels[varName] = varLabel.value
            if retcode:
                msg = "Problem getting variable label of variable %r" % varName
                checkErrsWarns(msg, retcode)
        return varLabels

    @varLabels.setter
    def varLabels(self, varLabels):
        if not varLabels:
            return
        func = self.spssio.spssSetVarLabel
        varLabels = self.encode(varLabels)
        for varName, varLabel in varLabels.items():
            retcode = func(c_int(self.fh), c_char_py3k(varName),
                           c_char_py3k(varLabel))
            if retcode:
                msg = ("Problem with setting variable label %r of variable %r"
                       % (varLabel, varName))
                checkErrsWarns(msg, retcode)

    @property
    @decode
    def formats(self):
        """Get the PRINT FORMATS, set PRINT and WRITE FORMATS.
        Returns/takes a dictionary of the form {varName: <format_>.
        For example: formats = {'salary': 'DOLLAR8', 'gender': 'A1',
                                'educ': 'F8.2'}"""
        if hasattr(self, "formats_"):
            return self.formats_
        printFormat_, printDec_, printWid_ = c_int(), c_int(), c_int()
        func = self.spssio.spssGetVarPrintFormat
        self.formats_ = {}
        for varName in self.varNames:
            vName = self.vNames[varName]
            retcode = func(c_int(self.fh), c_char_py3k(vName),
                           byref(printFormat_), byref(printDec_),
                           byref(printWid_))
            if retcode:
                msg = "Error getting print format for variable '%s'"
                checkErrsWarns(msg % vName.decode(), retcode)

            printFormat = allFormats.get(printFormat_.value)[0]
            printFormat = printFormat.split(b"_")[-1]
            format_ = printFormat + bytez(str(printWid_.value))
            if self.varTypes[varName] == 0:
                format_ += (b"." + bytez(str(printDec_.value)))
            if format_.endswith(b".0"):
                format_ = format_[:-2]
            self.formats_[varName] = format_
        return self.formats_

    def _splitformats(self):
        """This function returns the 'bare' formats + variable widths,
        e.g. format F5.3 is returned as 'F' and '5'"""
        pattern = b"(?P<bareFmt>[a-z]+)(?P<varWid>\d+)[.]?\d*"
        if self.ioUtf8_:
            pattern = pattern.decode("utf-8")
        regex = re.compile(pattern, re.I)
        bareformats, varWids = {}, {}
        for varName, format_ in self.formats.items():
            bareformat, varWid = regex.findall(format_)[0]
            bareformats[varName] = bareformat
            varWids[varName] = int(varWid)
        return bareformats, varWids

    @formats.setter
    def formats(self, formats):
        if not formats:
            return
        reverseFormats = dict([(v[0][9:], k) for k, v in allFormats.items()])
        validValues = sorted(reverseFormats.keys())
        regex = b"(?P<printFormat>A(HEX)?)(?P<printWid>\d+)"
        isStringVar = re.compile(regex, re.IGNORECASE)
        regex = b"(?P<printFormat>[A-Z]+)(?P<printWid>\d+)\.?(?P<printDec>\d*)"
        isAnyVar = re.compile(regex, re.IGNORECASE)
        funcP = self.spssio.spssSetVarPrintFormat  # print type
        funcW = self.spssio.spssSetVarWriteFormat  # write type
        for varName, format_ in self.encode(formats).items():
            format_ = format_.upper()
            gotString = isStringVar.match(format_)
            gotAny = isAnyVar.match(format_)
            msg = ("Unknown format %r or invalid width for variable %r. " +
                   "Valid formats are: %s")
            msg = msg % (format_, varName, b", ".join(validValues))
            if gotString:
                printFormat = gotString.group("printFormat")
                printFormat = reverseFormats.get(printFormat)
                printDec = 0
                printWid = int(gotString.group("printWid"))
            elif gotAny:
                printFormat = gotAny.group("printFormat")
                printFormat = reverseFormats.get(printFormat)
                printDec = gotAny.group("printDec")
                printDec = int(printDec) if printDec else 0
                printWid = int(gotAny.group("printWid"))
            else:
                raise ValueError(msg)

            if printFormat is None:
                raise ValueError(msg)

            args = (c_int(self.fh), c_char_py3k(varName), c_int(printFormat),
                    c_int(printDec), c_int(printWid))
            retcode1, retcode2 = funcP(*args), funcW(*args)
            if retcodes.get(retcode1) == "SPSS_INVALID_PRFOR":
                # invalid PRint FORmat
                msg = "format for %r misspecified (%r)"
                raise SPSSIOError(msg % (varName, format_), retcode1)
            if retcode1:
                msg = "Problem setting format_ %r for %r" % (format_, varName)
                checkErrsWarns(msg, retcode1)

    def _getMissingValue(self, varName):
        """This is a helper function for the missingValues getter
        method.  The function returns the missing values of variable <varName>
        as a a dictionary. The dictionary keys and items depend on the
        particular definition, which may be discrete values and/or ranges.
        Range definitions are only possible for numerical variables."""
        if self.varTypes[varName] == 0:
            func = self.spssio.spssGetVarNMissingValues
            args = (c_double(), c_double(), c_double())
        else:
            func = self.spssio.spssGetVarCMissingValues
            args = (create_string_buffer(9), create_string_buffer(9),
                    create_string_buffer(9))
        missingFmt = c_int()
        vName = self.vNames[varName]
        retcode = func(c_int(self.fh), c_char_py3k(vName),
                       byref(missingFmt), *map(byref, args))
        if retcode:
            msg = "Error getting missing value for variable '%s'" % varName
            checkErrsWarns(msg, retcode)

        v1, v2, v3 = [v.value for v in args]
        userMiss = dict([(v, k) for k, v in userMissingValues.items()])
        missingFmt = userMiss[missingFmt.value]
        if missingFmt == "SPSS_NO_MISSVAL":
            return {}
        elif missingFmt == "SPSS_ONE_MISSVAL":
            return {b"values": [v1]}
        elif missingFmt == "SPSS_TWO_MISSVAL":
            return {b"values": [v1, v2]}
        elif missingFmt == "SPSS_THREE_MISSVAL":
            return {b"values": [v1, v2, v3]}
        elif missingFmt == "SPSS_MISS_RANGE":
            return {b"lower": v1, b"upper": v2}
        elif missingFmt == "SPSS_MISS_RANGEANDVAL":
            return {b"lower": v1, b"upper": v2, b"value": v3}

    def _setMissingValue(self, varName, **kwargs):
        """This is a helper function for the missingValues setter
        method. The function sets missing values for variable <varName>.
        Valid keyword arguments are:
        * to specify a RANGE: 'lower', 'upper', optionally with 'value'
        * to specify DISCRETE VALUES: 'values', specified as a list no longer
        than three items, or as None, or as a float/int/str
        """
        if kwargs == {}:
            return 0
        fargs = ["lower", "upper", "value", "values"]
        if set(kwargs.keys()).difference(set(fargs)):
            raise ValueError("Allowed keywords are: %s" % ", ".join(fargs))
        varName = self.encode(varName)
        varType = self.varTypes[varName]

        # range of missing values, e.g. MISSING VALUES aNumVar (-9 THRU -1).
        if varType == 0:
            placeholder = 0.0
            if "lower" in kwargs and "upper" in kwargs and \
                "value" in kwargs:
                missingFmt = "SPSS_MISS_RANGEANDVAL"
                args = kwargs["lower"], kwargs["upper"], kwargs["value"]
            elif "lower" in kwargs and "upper" in kwargs:
                missingFmt = "SPSS_MISS_RANGE"
                args = kwargs["lower"], kwargs["upper"], placeholder
        else:
            placeholder, args = b"0", None

        # up to three discrete missing values
        if "values" in kwargs:
            values = self.encode(list(kwargs.values())[0])
            if isinstance(values, (float, int, str, bytes)):
                values = [values]

            # check if missing values strings values are not too long
            strMissLabels = [len(v) for v in values if 
                             isinstance(v, (str, bytes))]
            if strMissLabels and max(strMissLabels) > 9:
                raise ValueError("Missing value label > 9 bytes")

            nvalues = len(values) if values is not None else values
            if values is None or values == {}:
                missingFmt = "SPSS_NO_MISSVAL"
                args = placeholder, placeholder, placeholder
            elif nvalues == 1:
                missingFmt = "SPSS_ONE_MISSVAL"
                args = values + [placeholder, placeholder]
            elif nvalues == 2:
                missingFmt = "SPSS_TWO_MISSVAL"
                args = values + [placeholder]
            elif nvalues == 3:
                missingFmt = "SPSS_THREE_MISSVAL"
                args = values
            else:
                msg = "You can specify up to three individual missing values"
                raise ValueError(msg)

        # numerical vars
        if varType == 0 and args:
            func = self.spssio.spssSetVarNMissingValues
            func.argtypes = [c_int, c_char_p, c_int,
                             c_double, c_double, c_double]
            args = map(float, args)
        # string vars
        else:
            if args is None:
                raise ValueError("Illegal keyword for character variable")
            func = self.spssio.spssSetVarCMissingValues
            func.argtypes = [c_int, c_char_p, c_int,
                             c_char_p, c_char_p, c_char_p]

        retcode = func(self.fh, varName, userMissingValues[missingFmt], *args)
        if retcode:
            msg = "Problem setting missing value of variable %r" % varName
            checkErrsWarns(msg, retcode)

    @property
    @decode
    def missingValues(self):
        """Get/Set MISSING VALUES.
        User missing values are values that will not be included in
        calculations by SPSS. For example, 'don't know' might be coded as a
        user missing value (a value of 999 is typically used, so when vairable
        'age' has values 5, 15, 999, the average age is 10). This is
        different from 'system missing values', which are blank/null values.
        Takes a dictionary of the following form:
          {'someNumvar1': {'values': [999, -1, -2]}, # discrete values
           'someNumvar2': {'lower': -9, 'upper':-1}, # range "-9 THRU -1"
           'someNumvar3': {'lower': -9, 'upper':-1, 'value': 999},
           'someStrvar1': {'values': ['foo', 'bar', 'baz']},
           'someStrvar2': {'values': 'bletch'}}      # shorthand """
        missingValues = {}
        for varName in self.varNames:
            missingValues[varName] = self._getMissingValue(varName)
        return missingValues

    @missingValues.setter
    def missingValues(self, missingValues):
        if missingValues:
            for varName, kwargs in missingValues.items():
                self._setMissingValue(varName, **kwargs)

    # measurelevel, colwidth and alignment must all be set or not at all.
    @property
    @decode
    def measureLevels(self):
        """Get/Set VARIABLE LEVEL (measurement level).
        Returns/Takes a dictionary of the form {varName: varMeasureLevel}.
        Valid measurement levels are: "unknown", "nominal", "ordinal", "scale",
        "ratio", "flag", "typeless". This is used in Spss procedures such as
        CTABLES."""
        func = self.spssio.spssGetVarMeasureLevel
        levels = {0: b"unknown", 1: b"nominal", 2: b"ordinal", 3: b"scale",
                  3: b"ratio", 4: b"flag", 5: b"typeless"}
        measureLevel = c_int()
        varMeasureLevels = {}
        for varName in self.varNames:
            vName = self.vNames[varName]
            retcode = func(c_int(self.fh), c_char_py3k(vName),
                           byref(measureLevel))
            varMeasureLevels[varName] = levels.get(measureLevel.value)
            if retcode:
                msg = "Problem getting variable measurement level: %r"
                checkErrsWarns(msg % varName, retcode)

        return varMeasureLevels

    @measureLevels.setter
    def measureLevels(self, varMeasureLevels):
        if not varMeasureLevels:
            return
        func = self.spssio.spssSetVarMeasureLevel
        levels = {b"unknown": 0, b"nominal": 1, b"ordinal": 2, b"scale": 3,
                  b"ratio": 3, b"flag": 4, b"typeless": 5}
        for varName, level in self.encode(varMeasureLevels).items():
            if level.lower() not in levels:
                msg = "Valid levels are %s"
                raise ValueError(msg % b", ".join(levels.keys()).decode())
            level = levels.get(level.lower())
            retcode = func(c_int(self.fh), c_char_py3k(varName), c_int(level))
            if retcode:
                msg = "Problem setting variable mesasurement level: '%s'"
                checkErrsWarns(msg % varName.decode(), retcode)

    @property
    @decode
    def columnWidths(self):
        """Get/Set VARIABLE WIDTH (display width).
        Returns/Takes a dictionary of the form {varName: <int>}. A value of
        zero is special and means that the IBM SPSS Statistics Data Editor
        is to set an appropriate width using its own algorithm. If used,
        variable alignment, measurement level and column width all needs to
        be set."""
        func = self.spssio.spssGetVarColumnWidth
        varColumnWidth = c_int()
        varColumnWidths = {}
        for varName in self.varNames:
            vName = self.vNames[varName]
            retcode = func(c_int(self.fh), c_char_py3k(vName),
                           byref(varColumnWidth))
            if retcode:
                msg = "Problem getting column width: '%s'"
                checkErrsWarns(msg % varName.decode(), retcode)
            varColumnWidths[varName] = varColumnWidth.value
        return varColumnWidths

    @columnWidths.setter
    def columnWidths(self, varColumnWidths):
        if not varColumnWidths:
            return
        func = self.spssio.spssSetVarColumnWidth
        for varName, varColumnWidth in varColumnWidths.items():
            retcode = func(c_int(self.fh), c_char_py3k(varName),
                           c_int(varColumnWidth))
            if retcode:
                msg = "Error setting variable column width: '%s'"
                checkErrsWarns(msg % varName.decode(), retcode)

    def _setColWidth10(self):
        """Set the variable display width of string values to at least 10
        (it's annoying that SPSS displays e.g. a one-character variable in
        very narrow columns). This also sets all measurement levels to
        "unknown" and all variable alignments to "left". This function is
        only called if column widths, measurement levels and variable
        alignments are None."""
        columnWidths = {}
        for varName, varType in self.varTypes.items():
            # zero = appropriate width determined by spss
            columnWidths[varName] = 10 if 0 < varType < 10 else 0
        self.columnWidths = columnWidths
        self.measureLevels = dict([(v, b"unknown") for v in self.varNames])
        self.alignments = dict([(v, b"left") for v in self.varNames])

    @property
    @decode
    def alignments(self):
        """Get/Set VARIABLE ALIGNMENT.
        Returns/Takes a dictionary of the form {varName: alignment}
        Valid alignment values are: left, right, center. If used, variable
        alignment, measurement level and column width all need to be set.
        """
        func = self.spssio.spssGetVarAlignment
        alignments = {0: b"left", 1: b"right", 2: b"center"}
        alignment_ = c_int()
        varAlignments = {}
        for varName in self.varNames:
            vName = self.vNames[varName]
            retcode = func(c_int(self.fh), c_char_py3k(vName),
                           byref(alignment_))
            alignment = alignments[alignment_.value]
            varAlignments[varName] = alignment
            if retcode:
                msg = "Problem getting variable alignment: '%s'"
                checkErrsWarns(msg % varName.decode(), retcode)
        return varAlignments

    @alignments.setter
    def alignments(self, varAlignments):
        if not varAlignments:
            return
        func = self.spssio.spssSetVarAlignment
        alignments = {b"left": 0, b"right": 1, b"center": 2}
        for varName, varAlignment in varAlignments.items():
            if varAlignment.lower() not in alignments:
                ukeys = b", ".join(alignments.keys()).decode()
                raise ValueError("Valid alignments are %s" % ukeys)
            alignment = alignments.get(varAlignment.lower())
            retcode = func(c_int(self.fh), c_char_py3k(varName), c_int(alignment))
            if retcode:
                msg = "Problem setting variable alignment for variable '%s'"
                checkErrsWarns(msg % varName.decode(), retcode)

    @property
    @decode
    def varSets(self):
        """Get/Set VARIABLE SET information.
        Returns/Takes a dictionary with SETNAME as keys and a list of SPSS
        variables as values. For example: {'SALARY': ['salbegin',
        'salary'], 'DEMOGR': ['gender', 'minority', 'educ']}"""
        varSets = c_char_p()
        func = self.spssio.spssGetVariableSets
        retcode = func(c_int(self.fh), byref(varSets))
        if retcode:
            msg = "Problem getting variable set information"
            checkErrsWarns(msg, retcode)

        if not varSets.value:
            return {}
        varSets_ = {}
        for varSet in varSets.value.split(b"\n")[:-1]:
            k, v = varSet.split(b"= ")
            varSets_[k] = v.split()

        # clean up
        self.freeMemory("spssFreeVariableSets", varSets)

        return varSets_

    @varSets.setter
    def varSets(self, varSets):
        if not varSets:
            return
        varSets_ = []
        for varName, varSet in varSets.items():
            varName = varName.decode(self.fileEncoding)
            varSet = (b" ".join(varSet)).decode(self.fileEncoding)
            varSets_.append(("%s= %s" % (varName, varSet)).encode(self.fileEncoding))
        varSets_ = c_char_py3k(b"\n".join(varSets_))
        retcode = self.spssio.spssSetVariableSets(c_int(self.fh), varSets_)
        if retcode:
            msg = "Problem setting variable set information"
            checkErrsWarns(msg, retcode)

    @property
    @decode
    def varRoles(self):
        """Get/Set VARIABLE ROLES.
        Returns/Takes a dictionary of the form {varName: varRole}, where
        varRoles may be any of the following: 'both', 'frequency', 'input',
        'none', 'partition', 'record ID', 'split', 'target'"""
        func = self.spssio.spssGetVarRole
        roles = {0: b"input", 1: b"target", 2: b"both", 3: b"none", 4: b"partition",
                 5: b"split", 6: b"frequency", 7: b"record ID"}
        varRoles = {}
        varRole_ = c_int()
        for varName in self.varNames:
            vName = self.vNames[varName]
            retcode = func(c_int(self.fh), c_char_py3k(vName), byref(varRole_))
            varRole = roles.get(varRole_.value)
            varRoles[varName] = varRole
            if retcode:
                msg = "Problem getting variable role for variable %r"
                checkErrsWarns(msg, retcode)
        return varRoles

    @varRoles.setter
    def varRoles(self, varRoles):
        if not varRoles:
            return
        roles = {"input": 0, "target": 1, "both": 2, "none": 3, "partition": 4,
                 "split": 5,  "frequency": 6, "record ID": 7}
        func = self.spssio.spssSetVarRole
        for varName, varRole in varRoles.items():
            varRole = roles.get(varRole)
            retcode = func(c_int(self.fh), c_char_py3k(varName), c_int(varRole))
            if retcode:
                msg = "Problem setting variable role %r for variable %r"
                checkErrsWarns(msg % (varRole, varName), retcode)

    @property
    @decode
    def varAttributes(self):
        """Get/Set VARIABLE ATTRIBUTES.
        Returns/Takes dictionary of the form:
        {'var1': {'attr name x': 'attr value x','attr name y': 'attr value y'},
         'var2': {'attr name a': 'attr value a','attr name b': 'attr value b'}}
        """
        # abbreviation for readability and speed
        func = self.spssio.spssGetVarAttributes

        # initialize arrays
        MAX_ARRAY_SIZE = 1000
        attrNamesArr = (POINTER(c_char_p * MAX_ARRAY_SIZE))()
        attrValuesArr = (POINTER(c_char_p * MAX_ARRAY_SIZE))()

        attributes = {}
        for varName in self.varNames:
            vName = self.vNames[varName]

            # step 1: get array size
            nAttr = c_int()
            retcode = func(c_int(self.fh), c_char_py3k(vName),
                           byref(attrNamesArr), byref(attrValuesArr),
                           byref(nAttr))
            if retcode:
                msg = "Problem getting attributes of variable '%s' (step 1/2)"
                checkErrsWarns(msg % varName.decode(), retcode)

            # step 2: get attributes with arrays of proper size
            nAttr = c_int(nAttr.value)
            attrNamesArr = (POINTER(c_char_p * nAttr.value))()
            attrValuesArr = (POINTER(c_char_p * nAttr.value))()
            retcode = func(c_int(self.fh), c_char_py3k(vName),
                           byref(attrNamesArr), byref(attrValuesArr),
                           byref(nAttr))
            if retcode:
                msg = "Problem getting attributes of variable '%s' (step 2/2)"
                checkErrsWarns(msg % varName.decode(), retcode)

            # get array contents
            if not nAttr.value:
                continue
            k, v, n = attrNamesArr[0], attrValuesArr[0], nAttr.value
            attribute = dict([(k[i], v[i]) for i in xrange(n)])
            attributes[varName] = attribute

            # clean up
            args = (attrNamesArr, attrValuesArr, nAttr)
            self.freeMemory("spssFreeAttributes", *args)

        return attributes

    @varAttributes.setter
    def varAttributes(self, varAttributes):
        if not varAttributes:
            return
        func = self.spssio.spssSetVarAttributes
        for varName in self.varNames:
            attributes = varAttributes.get(varName)
            if not attributes:
                continue
            nAttr = len(attributes)
            attrNames = (c_char_p * nAttr)(*list(attributes.keys()))
            attrValues = (c_char_p * nAttr)(*list(attributes.values()))
            retcode = func(c_int(self.fh), c_char_py3k(varName),
                           pointer(attrNames), pointer(attrValues),
                           c_int(nAttr))
            if retcode:
                msg = "Problem setting variable attributes for variable %r"
                checkErrsWarns(msg % varName, retcode)

    @property
    @decode
    def fileAttributes(self):
        """Get/Set DATAFILE ATTRIBUTES.
        Returns/Takes a dictionary of the form:
        {'attrName[1]': 'attrValue1', 'revision[1]': '2010-10-09',
        'revision[2]': '2010-10-22', 'revision[3]': '2010-11-19'}
         """
        # abbreviation for readability
        func = self.spssio.spssGetFileAttributes

        # step 1: get array size
        MAX_ARRAY_SIZE = 100  # assume never more than 100 file attributes
        attrNamesArr = (POINTER(c_char_p * MAX_ARRAY_SIZE))()
        attrValuesArr = (POINTER(c_char_p * MAX_ARRAY_SIZE))()
        nAttr = c_int()
        retcode = func(c_int(self.fh), byref(attrNamesArr),
                       byref(attrValuesArr), byref(nAttr))

        # step 2: get attributes with arrays of proper size
        nAttr = c_int(nAttr.value)
        attrNamesArr = (POINTER(c_char_p * nAttr.value))()
        attrValuesArr = (POINTER(c_char_p * nAttr.value))()
        retcode = func(c_int(self.fh), byref(attrNamesArr),
                       byref(attrValuesArr), byref(nAttr))
        if retcode:
            checkErrsWarns("Problem getting file attributes", retcode)

        # get array contents
        if not nAttr.value:
            return {}
        k, v = attrNamesArr[0], attrValuesArr[0]
        attributes = dict([(k[i], v[i]) for i in xrange(nAttr.value)])

        # clean up
        args = (attrNamesArr, attrValuesArr, nAttr)
        self.freeMemory("spssFreeAttributes", *args)

        return attributes

    @fileAttributes.setter
    def fileAttributes(self, fileAttributes):
        if not fileAttributes:
            return
        nAttr = len(fileAttributes)
        attrNames = (c_char_p * nAttr)(*list(fileAttributes.keys()))
        attrValues = (c_char_p * nAttr)(*list(fileAttributes.values()))
        func = self.spssio.spssSetFileAttributes
        retcode = func(c_int(self.fh), byref(attrNames),
                       byref(attrValues), c_int(nAttr))
        if retcode:
            checkErrsWarns("Problem setting file attributes", retcode)

    def _getMultRespDef(self, mrDef):
        """Get 'normal' multiple response defintions.
        This is a helper function for the multRespDefs getter function.
        A multiple response definition <mrDef> in the string format returned
        by the IO module is converted into a multiple response definition of
        the form multRespSet = {<setName>: {"setType": <setType>, "label":
        <lbl>, "varNames": <list_of_varNames>}}. SetType may be either 'D'
        (multiple dichotomy sets) or 'C' (multiple category sets). If setType
        is 'D', the multiple response definition also includes '"countedValue":
        countedValue'"""
        regex = b"\$(?P<setName>\S+)=(?P<setType>[CD])\n?"
        m = re.search(regex + b".*", mrDef, re.I | re.L)
        if not m:
            return {}
        setType = m.group("setType")
        if setType == b"C":  # multiple category sets
            regex += b" (?P<lblLen>\d+) (?P<lblVarNames>.+) ?\n?"
            matches = re.findall(regex, mrDef, re.I)
            setName, setType, lblLen, lblVarNames = matches[0]
        else:               # multiple dichotomy sets
            # \w+ won't always work (e.g. thai) --> \S+
            regex += (b"(?P<valueLen>\d+) (?P<countedValue>\S+)" +
                      b" (?P<lblLen>\d+) (?P<lblVarNames>.+) ?\n?")
            matches = re.findall(regex, mrDef, re.I | re.L)
            setName, setType, valueLen = matches[0][:3]
            countedValue, lblLen, lblVarNames = matches[0][3:]
        lbl = lblVarNames[:int(lblLen)]
        varNames = lblVarNames[int(lblLen):].split()
        multRespSet = {setName: {b"setType": setType, b"label": lbl,
                                 b"varNames": varNames}}
        if setType == b"D":
            multRespSet[setName][b"countedValue"] = countedValue
        return multRespSet

    def _setMultRespDefs(self, multRespDefs):
        """Set 'normal' multiple response defintions.
        This is a helper function for the multRespDefs setter function. 
        It translates the multiple response definition, specified as a
        dictionary, into a string that the IO module can use"""
        mrespDefs = []
        for setName, rest in multRespDefs.items():
            rest = self.encode(rest)
            if rest["setType"] not in (b"C", b"D"):
                continue
            rest["setName"] = self.encode(setName)
            mrespDef = b"$%(setName)s=%(setType)s" % rest
            lblLen = len(rest["label"])
            rest["lblLen"] = lblLen
            rest["varNames"] = b" ".join(rest["varNames"])
            tail = b" %(varNames)s" if lblLen == 0 else b"%(label)s %(varNames)s"
            if rest["setType"] == b"C":  # multiple category sets
                template = b" %%(lblLen)s %s " % tail
            else:                       # multiple dichotomy sets
                # line below added/modified after Issue #4:
                # Assertion during creating of multRespDefs
                rest["valueLen"] = len(str(rest["countedValue"]))
                template = (b"%%(valueLen)s %%(countedValue)s %%(lblLen)s %s "
                            % tail)
            mrespDef += template % rest
            mrespDefs.append(mrespDef.rstrip())
        mrespDefs = b"\n".join(mrespDefs)
        return mrespDefs

    def _getMultRespDefsEx(self, mrDef):
        """Get 'extended' multiple response defintions.
        This is a helper function for the multRespDefs getter function."""
        regex = ("\$(?P<setName>\w+)=(?P<setType>E) (?P<flag1>1)" +
                 "(?P<flag2>1)? (?P<valueLen>[0-9]+) (?P<countedValue>\w+) " +
                 "(?P<lblLen>[0-9]+) (?P<lblVarNames>[\w ]+)")
        matches = re.findall(regex, mrDef, re.I | re.U)
        setName, setType, flag1, flag2 = matches[0][:4]
        valueLen, countedValue, lblLen, lblVarNames = matches[0][4:]
        length = int(lblLen)
        label, varNames = lblVarNames[:length], lblVarNames[length:].split()
        return {setName: {"setType": setType, "firstVarIsLabel": bool(flag2),
                          "label": label, "countedValue": countedValue,
                          "varNames": varNames}}

    @property
    @decode
    def multRespDefs(self):
        """Get/Set MRSETS (multiple response) sets.
        Returns/takes a dictionary of the form:
        --multiple category sets: {setName: {"setType": "C", "label": lbl,
          "varNames": [<list_of_varNames>]}}
        --multiple dichotomy sets: {setName: {"setType": "D", "label": lbl,
          "varNames": [<list_of_varNames>], "countedValue": countedValue}}
        --extended multiple dichotomy sets: {setName: {"setType": "E",
          "label": lbl, "varNames": [<list_of_varNames>], "countedValue":
           countedValue, 'firstVarIsLabel': <bool>}}
	    Note. You can get values of extended multiple dichotomy sets with
        getMultRespSetsDefEx, but you cannot write extended multiple dichotomy
        sets.

        For example:
        categorical  = {"setType": "C", "label": "labelC",
                       "varNames": ["salary", "educ"]}
        dichotomous1 = {"setType": "D", "label": "labelD",
                        "varNames": ["salary", "educ"], "countedValue": "Yes"}
        dichotomous2 = {"setType": "D", "label": "", "varNames":
                        ["salary", "educ", "jobcat"], "countedValue": "No"}
        extended1    = {"setType": "E", "label": "", "varNames": ["mevar1",
                        "mevar2", "mevar3"], "countedValue": "1",
                        "firstVarIsLabel": True}
        extended2    = {"setType": "E", "label":
                        "Enhanced set with user specified label", "varNames":
                        ["mevar4", "mevar5", "mevar6"], "countedValue":
                        "Yes", "firstVarIsLabel": False}
        multRespDefs = {"testSetC": categorical, "testSetD1": dichotomous1,
                        "testSetD2": dichotomous2, "testSetEx1": extended1,
                        "testSetEx2": extended2}
        """
        #####
        # I am not sure whether 'extended' MR definitions complement
        # or replace 'normal' MR definitions. I assumed 'complement'.
        #####

        ## Normal Multiple response definitions
        func = self.spssio.spssGetMultRespDefs
        mrDefs = c_char_p()
        retcode = func(c_int(self.fh), pointer(mrDefs))
        if retcode:
            msg = "Problem getting multiple response definitions"
            checkErrsWarns(msg, retcode)

        multRespDefs = {}
        if mrDefs.value:
            for mrDef in mrDefs.value.split(b"\n"):
                for setName, rest in self._getMultRespDef(mrDef).items():
                    multRespDefs[setName] = rest
            self.freeMemory("spssFreeMultRespDefs", mrDefs)

        ## Extended Multiple response definitions
        mrDefsEx = c_char_p()
        func = self.spssio.spssGetMultRespDefsEx
        retcode = func(c_int(self.fh), pointer(mrDefsEx))
        if retcode:
            msg = "Problem getting extended multiple response definitions"
            checkErrsWarns(msg, retcode)

        multRespDefsEx = {}
        if mrDefsEx.value:
            for mrDefEx in mrDefsEx.value.split(b"\n"):
                for setName, rest in self._getMultRespDef(mrDefEx).items():
                    multRespDefsEx[setName] = rest
            self.freeMemory("spssFreeMultRespDefs", mrDefsEx)

        multRespDefs.update(multRespDefsEx)
        return multRespDefs

    @multRespDefs.setter
    def multRespDefs(self, multRespDefs):
        if not multRespDefs:
            return
        multRespDefs = self._setMultRespDefs(multRespDefs)
        func = self.spssio.spssSetMultRespDefs
        retcode = func(c_int(self.fh), c_char_py3k(multRespDefs))
        if retcode:
            msg = "Problem setting multiple response definitions"
            checkErrsWarns(msg, retcode)

    @property
    @decode
    def caseWeightVar(self):
        """Get/Set WEIGHT variable.
        Takes a valid varName, and returns weight variable, if any, as a
        string."""
        varNameBuff = create_string_buffer(65)
        func = self.spssio.spssGetCaseWeightVar
        retcode = func(c_int(self.fh), byref(varNameBuff))
        if retcode:
            msg = "Problem getting case weight variable name"
            checkErrsWarns(msg, retcode)
        return varNameBuff.value

    @caseWeightVar.setter
    def caseWeightVar(self, varName):
        if not varName:
            return
        func = self.spssio.spssSetCaseWeightVar
        retcode = func(c_int(self.fh), c_char_py3k(varName))
        if retcode:
            msg = "Problem setting case weight variable name %r" % varName
            checkErrsWarns(msg, retcode)

    @property
    @decode
    def dateVariables(self):  # seems to be okay
        """Get/Set DATE information. This function reports the Forecasting
        (Trends) date variable information, if any, in IBM SPSS Statistics
        data files. Entirely untested and not implemented in reader/writer"""
        # step 1: get array size
        nElements = c_int()
        func = self.spssio.spssGetDateVariables
        MAX_ARRAY_SIZE = 100
        dateInfoArr = (POINTER(c_long * MAX_ARRAY_SIZE))()
        retcode = func(c_int(self.fh), byref(nElements), byref(dateInfoArr))

        # step 2: get date info with array of proper size
        dateInfoArr = (POINTER(c_long * nElements.value))()
        retcode = func(c_int(self.fh), byref(nElements), byref(dateInfoArr))
        if retcode:
            checkErrsWarns("Problem getting TRENDS information", retcode)

        # get array contents
        nElem = nElements.value
        if not nElem:
            return {}
        dateInfo = [dateInfoArr[0][i] for i in xrange(nElem)]
        fixedDateInfo = dateInfo[:6]
        otherDateInfo = [dateInfo[i: i + 3] for i in xrange(6, nElem, 3)]
        dateInfo = {"fixedDateInfo": fixedDateInfo,
                    "otherDateInfo": otherDateInfo}

        # clean up
        self.freeMemory("spssFreeDateVariables", dateInfoArr)

        return dateInfo

    @dateVariables.setter
    def dateVariables(self, dateInfo):  # 'SPSS_INVALID_DATEINFO'!
        dateInfo = [dateInfo["fixedDateInfo"]] + dateInfo["otherDateInfo"]
        dateInfo = reduce(list.__add__, dateInfo)  # flatten list
        isAllInts = all([isinstance(d, int) for d in dateInfo])
        isSixPlusTriplets = (len(dateInfo) - 6) % 3 == 0
        if not isAllInts and isSixPlusTriplets:
            msg = ("TRENDS date info must consist of 6 fixed elements"
                   "+ <nCases> three-element groups of other date info "
                   "(all ints)")
            raise TypeError(msg)
        func = self.spssio.spssSetDateVariables
        nElements = len(dateInfo)
        dateInfoArr = (c_long * nElements)(*dateInfo)
        retcode = func(c_int(self.fh), c_int(nElements), dateInfoArr)
        if retcode:
            checkErrsWarns("Problem setting TRENDS information", retcode)

    @property
    @decode
    def textInfo(self):
        """Get/Set text information.
        Takes a savFileName and returns a string of the form: "File %r built
        using SavReaderWriter.py version %s (%s)". This is akin to, but
        *not* equivalent to the SPSS syntax command DISPLAY DOCUMENTS"""
        textInfo = create_string_buffer(256)
        retcode = self.spssio.spssGetTextInfo(c_int(self.fh), byref(textInfo))
        if retcode:
            checkErrsWarns("Problem getting textInfo", retcode)
        return textInfo.value

    @textInfo.setter
    def textInfo(self, savFileName):
        info = (os.path.basename(savFileName), __version__, time.asctime())
        textInfo = "File '%s' built using savReaderWriter version %s (%s)"
        textInfo = textInfo % info
        if self.ioUtf8 and isinstance(savFileName, unicode):
            textInfo = textInfo.encode("utf-8")
        func = self.spssio.spssSetTextInfo
        retcode = func(c_int(self.fh), c_char_py3k(textInfo[:256]))
        if retcode:
            checkErrsWarns("Problem setting textInfo", retcode)

    @property
    @decode
    def fileLabel(self):
        """Get/Set FILE LABEL (id string)
        Takes a file label (basestring), and returns file label, if any, as
        a string."""
        idStr = create_string_buffer(65)
        retcode = self.spssio.spssGetIdString(c_int(self.fh), byref(idStr))
        if retcode:
            checkErrsWarns("Error getting file label (id string)", retcode)
        return idStr.value

    @fileLabel.setter
    def fileLabel(self, idStr):
        if idStr is None:
            idStr = ("File created by user %r at %s"[:64] %
                     (getpass.getuser(), time.asctime()))
        if self.ioUtf8 and isinstance(idStr, unicode):
            idStr = idStr.encode("utf-8")
        retcode = self.spssio.spssSetIdString(c_int(self.fh), c_char_py3k(idStr))
        if retcode:
            checkErrsWarns("Problem setting file label (id string)", retcode)

    @property
    def queryType7(self):
        """This function can be used to determine whether a file opened for reading
        or append contains a specific "type 7" record. Returns a dictionary of the
        form: {subtype_number: (subtype_label, present_or_not)}, where
        present_or_not is a bool"""
        subtypes = \
                 {3: "Release information",
                  4: "Floating point constants including the system missing value",
                  5: "Variable set definitions",
                  6: "Date variable information",
                  7: "Multiple-response set definitions",
                  8: "Data Entry for Windows (DEW) information",
                 10: "TextSmart information",
                 11: ("Measurement level, column width, and " +
                      "alignment for each variable")}
        type7info = {}
        for subtype, label in subtypes.items():
            bFound = c_int()
            args = c_int(self.fh), c_int(subtype), byref(bFound)
            retcode = self.spssio.spssQueryType7(*args)
            if retcode:
                checkErrsWarns("Problem retrieving type7 info", retcode)
            type7info[subtype] = (label, bool(bFound.value))
        return type7info

    @property
    def dataEntryInfo(self):
        """Get/Set information that is private to the Data Entry for Windows (DEW)
        product. Returns/takes a dictionary of the form:
        dataEntryInfo = {"data": [<list_of_dew_segments>], "GUID": <guid>},
        where GUID stands for 'globally unique identifier'. 
        Some remarks:
        -A difference in the byte order of the host system and the foreign host
         will result in an error. Therefore, an optional 'swapBytes' key may 
         be specified whose value indicates whether the bytes should be swapped 
         (True) or not (False). Default is that the byte order of the host system
         is retained.
        -DEW information is not copied when using mode="cp" in the SavWriter
         initializer
        -THIS IS ENTIRELY UNTESTED!"""
        # check if file and host system byte order match
        # spssGetDEWInfo will return SPSS_NO_DEW, which is less desirable
        endianness = self.releaseInfo["big/little-endian code"]
        file_byte_order = 'little' if endianness == 0 else 'big'
        if file_byte_order != sys.byteorder:
            msg = "Host (%s-endian) and file (%s-endian) byte order differ"
            raise ValueError(msg % (sys.byteorder, file_byte_order))

        # retrieve length of DEW information (in bytes)
        pLength, pHashTotal = c_long(), c_long()
        func = self.spssio.spssGetDEWInfo
        args = c_int(self.fh), byref(pLength), byref(pHashTotal)
        retcode = func(*args)
        maxData = pLength.value  # Maximum bytes to return
        if not maxData:
            return {}  # file contains no DEW info

        # retrieve first segment of DEW information
        if not retcode:
            nData, pData = c_long(), c_void_p()
            func =  self.spssio.spssGetDEWFirst
            args = c_int(self.fh), byref(pData), c_long(maxData), byref(nData)
            retcode = func(*args)
            dew_information = [pData.value]

        # retrieve subsequent segments of DEW information
        if not retcode:
            func = self.spssio.spssGetDEWNext
            for i in range(nData.value - 1):
                nData = c_long()
                retcode = func(*args)
                if retcode > 0:
                    break
                dew_information.append(pData.value)

        # retieve GUID information
        if not retcode:
            func = self.spssio.spssGetDEWGUID
            asciiGUID = create_string_buffer(257)
            retcode = func(c_int(self.fh), byref(asciiGUID))

        if retcode:
            msg = "Problem getting Data Entry info with function %r"
            checkErrsWarns(msg % func.__name__, retcode)
        return dict(data=dew_information, GUID=asciiGUID.value)

    @dataEntryInfo.setter
    def dataEntryInfo(self, info):
        data, asciiGUID = info["data"], info["GUID"]
        # input validation
        is_ascii = all(map(lambda x: ord(x) < 128, asciiGUID))
        if not isinstance(asciiGUID, str) and is_ascii:
            raise ValueError("GUID must be a string of ascii characters")
        
        # I am not sure at all about the following
        swapit = info.has_key("swapBytes") and info.get("swapBytes")
        def swap(x):
           """swap bytes if needed"""
           src_fmt = '<%s' if sys.byteorder == 'little' else '>%s'
           dst_fmt = ">%s" if swapit and src_fmt[0] == "<" else "<%s"
           if isinstance(x, (float, int)):
               src_fmt, dst_fmt = src_fmt % "l", dst_mft % "l"
           elif isinstance(x, str):
               src_fmt, dst_fmt = src_fmt % "s", dst_mft % "s"
           else:
               type_ = re.search("'(\w+)'", str(type(x))).group(1)
               raise TypeError("Must be str, int or float, not %s") % type_
           if src_fmt != dst_fmt:
               x = struct.unpack(dst_fmt, struct.pack(src_fmt, x))[0]
           return x
        if swapit:
            data, asciiGUID = map(swap, data), swap(asciiGUID)

        # write DEW information
        for i, pData in enumerate(data):
            nBytes = len(pData)
            args = c_int(self.fh), c_void_p(pData), c_long(nBytes)
            # ... first segment
            if not i:
                func = self.spssio.spssSetDEWFirst
                retcode = func(*args)
            # ... subsequent segments
            else:
                func = self.spssio.spssSetDEWNext
                retcode = func(*args)
            if retcode > 0:
                break

        # write GUI information
        if not retcode:
            args = c_int(self.fh), c_char_py3k(asciiGUID)
            func = self.spssio.spssSetDEWGUID
            retcode = func(*args)
        else:
            msg = "Problem setting Data Entry info with function %r"
            checkErrsWarns(msg % func.__name__, retcode)
