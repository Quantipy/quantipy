#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ctypes import *
import os
import operator
import locale
import datetime
import collections

from savReaderWriter import *
from header import *

@rich_comparison
@implements_to_string
class SavReader(Header):
    """ Read Spss system files (.sav, .zsav)

    Parameters:
    -savFileName: the file name of the spss data file
    -returnHeader: Boolean that indicates whether the first record should
        be a list of variable names (default = False)
    -recodeSysmisTo: indicates to which value missing values should
        be recoded (default = None),
    -selectVars: indicates which variables in the file should be selected.
        The variables should be specified as a list or a tuple of
        valid variable names. If None is specified, all variables
        in the file are used (default = None)
    -idVar: indicates which variable in the file should be used for use as id
        variable for the 'get' method (default = None)
    -verbose: Boolean that indicates whether information about the spss data
        file (e.g., number of cases, variable names, file size) should be
        printed on the screen (default = False).
    -rawMode: Boolean that indicates whether values should get SPSS-style
        formatting, and whether date variables (if present) should be converted
        to ISO-dates. If True, the program does not format any values, which
        increases processing speed. (default = False)
    -ioUtf8: indicates the mode in which text communicated to or from the
        I/O Module will be. Valid values are True (UTF-8 mode aka Unicode mode)
        and False (Codepage mode). Cf. SET UNICODE=ON/OFF (default = False)
    -ioLocale: indicates the locale of the I/O module. Cf. SET LOCALE (default
        = None, which corresponds to ".".join(locale.getlocale())

    Typical use:
    savFileName = "someFile.sav"
    with SavReader(savFileName, returnHeader=True) as reader:
        header = next(reader)
        for line in reader:
            process(line)
    """

    def __init__(self, savFileName, returnHeader=False, recodeSysmisTo=None,
                 verbose=False, selectVars=None, idVar=None, rawMode=False,
                 ioUtf8=False, ioLocale=None):
        """ Constructor. Initializes all vars that can be recycled """
        super(SavReader, self).__init__(savFileName, b"rb", None,
                                        ioUtf8, ioLocale)
        self.savFileName = savFileName
        self.returnHeader = returnHeader
        self.recodeSysmisTo = recodeSysmisTo
        self.verbose = verbose
        self.selectVars = selectVars
        self.idVar = idVar
        self.rawMode = rawMode

        self.header = self.getHeader(self.selectVars)
        self.bareformats, self.varWids = self._splitformats()
        self.autoRawMode = self._isAutoRawMode()

        self.ioUtf8_ = ioUtf8
        self.sysmis_ = self.sysmis
        self.numVars = self.numberofVariables
        self.nCases = self.numberofCases

        self.myStruct = self.getStruct(self.varTypes, self.varNames)
        self.unpack_from = self.myStruct.unpack_from
        self.seekNextCase = self.spssio.spssSeekNextCase
        self.caseBuffer = self.getCaseBuffer()

        if psycoOk:
            self._items = psyco.proxy(self._items)  # 3 x faster!

    def __enter__(self):
        """ This function opens the spss data file (context manager)."""
        if self.verbose and self.ioUtf8_:
            print(self.replace(os.linesep, "\n"))
        elif self.verbose:
            print(str(self).replace(os.linesep, "\n"))
        return iter(self)

    def __exit__(self, type, value, tb):
        """ This function closes the spss data file and does some cleaning."""
        if type is not None:
            pass  # Exception occurred
        self.close()

    def close(self):
        """This function closes the spss data file and does some cleaning."""
        if not segfaults:
            self.closeSavFile(self.fh, mode=b"rb")
        del self.spssio

    def __len__(self):
        """ This function reports the number of cases (rows) in the spss data
        file. For example: len(SavReader(savFileName))"""
        return self.nCases

    # Python 3: see @rich_comparison class decorator
    def __cmp__(self, other):
        """ This function implements behavior for all of the comparison
        operators so comparisons can be made between SavReader instances,
        or comparisons between SavReader instances and integers."""
        if not isinstance(other, (SavReader, int)):
            raise TypeError
        other = other if isinstance(other, int) else len(other)
        if len(self) < other:
            return -1
        elif len(self) == other:
            return 0
        else:
            return 1

    def __hash__(self):
        """This function returns a hash value for the object to ensure it
        is hashable."""
        return id(self)

    def __str__(self):
        """This function returns a conscise file report of the spss data file
        For example str(SavReader(savFileName))"""
        return self.__unicode__().encode(self.fileEncoding)

    def __unicode__(self):
        """This function returns a conscise file report of the spss data file,
        For example unicode(SavReader(savFileName))"""
        return self.getFileReport()

    @property
    def shape(self):
        """This function returns the number of rows (nrows) and columns
        (ncols) as a namedtuple"""
        dim = (self.nCases, self.numVars)
        return collections.namedtuple("_", "nrows ncols")(*dim)

    def _isAutoRawMode(self):
        """Helper function for formatValues function. Determines whether
        iterating over each individual value is really needed"""
        hasDates = bool(set(self.bareformats.values()) & set(supportedDates))
        hasNfmt = b"N" in self.bareformats
        hasRecodeSysmis = self.recodeSysmisTo is not None
        items = [hasDates, hasNfmt, hasRecodeSysmis, self.ioUtf8_]
        return False if any(items) else True

    def formatValues(self, record):
        """This function formats date fields to ISO dates (yyyy-mm-dd), plus
        some other date/time formats. The SPSS N format is formatted to a
        character value with leading zeroes. System missing values are recoded
        to <recodeSysmisTo>. If rawMode==True, this function does nothing"""
        if self.rawMode or self.autoRawMode:
            return record  # 6-7 times faster!
        for i, value in enumerate(record):
            varName = self.header[i]
            varType = self.varTypes[varName]
            bareformat_ = self.bareformats[varName]
            varWid = self.varWids[varName]
            if varType == 0:
                # recode system missing values, if present and desired
                if value > self.sysmis_:
                    pass
                else:
                    record[i] = self.recodeSysmisTo
                # format N-type values (=numerical with leading zeroes)
                if bareformat_ == "N":
                    #record[i] = str(value).zfill(varWid)
                    record[i] = "%%0%dd" % varWid % value  #15 x faster (zfill)
                # convert SPSS dates to ISO dates
                elif bareformat_ in supportedDates:
                    fmt = supportedDates[bareformat_]
                    args = (value, fmt, self.recodeSysmisTo)
                    record[i] = self.spss2strDate(*args)
                    if bareformat_ == b"QYR" and record[i]:
                        # convert month to quarter, e.g. 12 Q 1990 --> 4 Q 1990
                        # There is no such thing as a %q strftime directive
                        try:
                            record[i] = QUARTERS[record[i][:2]] + record[i][2:]
                        except (KeyError, TypeError):
                            record[i] = self.recodeSysmisTo
            elif varType > 0:
                value = value[:varType]
                if self.ioUtf8_:
                    record[i] = value.decode("utf-8")
                else:
                    record[i] = value
        return record

    def _items(self, start=0, stop=None, step=1, returnHeader=False):
        """ This is a helper function to implement the __getitem__ and
        the __iter__ special methods. """

        if returnHeader:
            yield self.header

        if all([start == 0, stop is None, step == 1]):  # used as iterator
            retcode = 0
        else:
            retcode = self.seekNextCase(c_int(self.fh), c_long(0))  # reset

        if retcode:
            checkErrsWarns("Problem seeking first case", retcode)

        if stop is None:
            stop = self.nCases

        selection = self.selectVars is not None
        for case in xrange(start, stop, step):
            if start:
                # only call this when iterating over part of the records
                retcode = self.seekNextCase(c_int(self.fh), c_long(case))
                if retcode:
                    checkErrsWarns("Problem seeking case %d" % case, retcode)
            elif stop == self.nCases:
                self.printPctProgress(case, self.nCases)

            record = self.record

            if selection:
                record = self.selector(record)
                record = [record] if len(record) == 1 else list(record)
            record = self.formatValues(record)
            yield record

    def __iter__(self):
        """This function allows the object to be used as an iterator"""
        return self._items(0, None, 1, self.returnHeader)

    def __getitem__(self, key):
        """ This function reports the record of case number <key>.
        For example: firstRecord = SavReader(savFileName)[0].
        The <key> argument may also be a slice, for example:
        firstfiveRecords = SavReader(savFileName)[:5].
        You can also do stuff like (numpy required!):
        savReader(savFileName)[1:5, 1]"""

        is_slice = isinstance(key, slice)
        is_array_slice = key is Ellipsis or isinstance(key, tuple)

        if is_slice:
            start, stop, step = key.indices(self.nCases)
        elif is_array_slice:
            return self._get_array_slice(key, self.nCases, len(self.header))
        else:
            key = operator.index(key)
            start = key + self.nCases if key < 0 else key
            if not 0 <= start < self.nCases:
                raise IndexError("Index out of bounds")
            stop = start + 1
            step = 1

        records = self._items(start, stop, step)
        if is_slice:
            return list(records)
        return next(records)

    def _get_array_slice(self, key, nRows, nCols):
        """This is a helper function to implement array slicing with numpy"""
        if not numpyOk:
            raise ImportError("Array slicing requires the numpy library")

        is_index = False
        rstart = cstart = 0
        cstop = cstep = None
        try:
            row, col = key
            if row < 0:
                row = nRows + row
            if col < 0:
                col = nCols + col

            ## ... slices
            if isinstance(row, slice) and col is Ellipsis:
                # reader[1:2, ...]
                rstart, rstop, rstep = row.indices(nRows)
                cstart, cstop, cstep = 0, nRows, 1
            elif row is Ellipsis and isinstance(col, slice):
                # reader[..., 1:2]
                rstart, rstop, rstep = 0, nRows, 1
                cstart, cstop, cstep = col.indices(nCols)
            elif isinstance(row, slice) and isinstance(col, slice):
                # reader[1:2, 1:2]
                rstart, rstop, rstep = row.indices(nRows)
                cstart, cstop, cstep = col.indices(nCols)
            elif row is Ellipsis and col is Ellipsis:
                # reader[..., ...]
                rstart, rstop, rstep = 0, nRows, 1
                cstart, cstop, cstep = 0, nCols, 1

            ## ... indexes
            elif isinstance(row, int) and col is Ellipsis:
                # reader[1, ...]
                rstart, rstop, rstep = row, row + 1, 1
                cstart, cstop, cstep = 0, nCols, 1
                is_index = True
            elif row is Ellipsis and isinstance(col, int):
                # reader[..., 1]
                rstart, rstop, rstep = 0, nRows, 1
                cstart, cstop, cstep = col, col + 1, 1
                is_index = True
            elif  isinstance(row, int) and isinstance(col, int):
                # reader[1, 1]
                rstart, rstop, rstep = row, row + 1, 1
                cstart, cstop, cstep = col, col + 1, 1
                is_index = True

            # ... slice + index
            elif isinstance(row, slice) and isinstance(col, int):
                # reader[1:2, 1]
                rstart, rstop, rstep = row.indices(nRows)
                cstart, cstop, cstep = col, col + 1, 1
            elif isinstance(row, int) and isinstance(col, slice):
                # reader[1, 1:2]
                rstart, rstop, rstep = row, row + 1, 1
                cstart, cstop, cstep = col.indices(nCols)

            try:
                if not 0 <= rstart < nRows:
                    raise IndexError("Index out of bounds")
                if not 0 <= cstart < nCols:
                    raise IndexError("Index out of bounds")
                key = (Ellipsis, slice(cstart, cstop, cstep))

            except UnboundLocalError:
                msg = "The array index is either invalid, or not implemented"
                raise TypeError(msg)

        except TypeError:
            # reader[...]
            rstart, rstop, rstep = 0, nRows, 1
            key = (Ellipsis, Ellipsis)
        records = self._items(rstart, rstop, rstep)
        result = numpy.array(list(records))[key].tolist()
        if abs(key[1].start - key[1].stop) == 1:
            return reduce(list.__add__, result) # flatten list if it's one col
        if is_index:
            return result[0]
        return result

    def head(self, n=5):
        """ This convenience function returns the first <n> records. """
        return self[:abs(n)]

    def tail(self, n=5):
        """ This convenience function returns the last <n> records. """
        return self[-abs(n):]

    def all(self):
        """ This convenience function returns all the records. """        
        return [record for record in iter(self)]

    def __contains__(self, item):
        """ This function implements membership testing and returns True if
        <idVar> contains <item>. Thus, it requires the 'idVar' parameter to
        be set. For example: reader = SavReader(savFileName, idVar="ssn")
        "987654321" in reader """
        return bool(self.get(item))

    def get(self, key, default=None, full=False):
        """ This function returns the records for which <idVar> == <key>
        if <key> in <savFileName>, else <default>. Thus, the function mimics
        dict.get, but note that dict[key] is NOT implemented. NB: Even though
        this uses a binary search, this is not very fast on large data (esp.
        the first call, and with full=True)

        Parameters:
        -key: key for which the corresponding record should be returned
        -default: value that should be returned if <key> is not found
          (default: None)
        -full: value that indicates whether *all* records for which
          <idVar> == <key> should be returned (default: False)
        For example: reader = SavReader(savFileName, idVar="ssn")
        reader.get("987654321", "social security number not found!")"""

        if not self.idVar in self.varNames:
            msg = ("SavReader object must be instantiated with an existing " +
                   "variable as an idVar argument")
            raise NameError(msg)

        #two slightly modified functions from the bisect module
        def bisect_right(a, x, lo=0, hi=None):
            if hi is None:
                hi = len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if x < a[mid][0]:
                    hi = mid  # a[mid][0], not a[mid]
                else:
                    lo = mid + 1
            return lo

        def bisect_left(a, x, lo=0, hi=None):
            if hi is None:
                hi = len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if a[mid][0] < x:
                    lo = mid + 1  # a[mid][0], not a[mid]
                else:
                    hi = mid
            return lo

        idPos = self.varNames.index(self.idVar)
        if not hasattr(self, "isSorted"):
            self.isSorted = True
            if self.varTypes[self.idVar] == 0:
                if not isinstance(key, (int, float)):
                    return default
                self.recordz = ((record[idPos], i) for i,
                                record in enumerate(iter(self)))
            else:
                if not isinstance(key, basestring):
                    return default
                self.recordz = ((record[idPos].rstrip(), i) for i,
                                record in enumerate(iter(self)))
            self.recordz = sorted(self.recordz)
        insertLPos = bisect_left(self.recordz, key)
        insertRPos = bisect_right(self.recordz, key)
        if full:
            result = [self[record[1]] for record in
                      self.recordz[insertLPos: insertRPos]]
        else:
            if insertLPos == insertRPos:
                return default
            result = self[self.recordz[insertLPos][1]]
        if result:
            return result
        return default

    def getSavFileInfo(self):
        """ This function reads and returns some basic information of the open
        spss data file."""
        return (self.numVars, self.nCases, self.varNames, self.varTypes,
                self.formats, self.varLabels, self.valueLabels)

    def memoize(f):
        """Memoization decorator
        http://code.activestate.com/recipes/577219-minimalistic-memoization/"""
        cache = {}
        MAXCACHE = 10**4

        def memf(*x):
            if x not in cache:
                cache[x] = f(*x)
            if len(cache) > MAXCACHE:
                cache.popitem()
            return cache[x]
        return memf

    @memoize
    def spss2strDate(self, spssDateValue, fmt, recodeSysmisTo):
        """This function converts internal SPSS dates (number of seconds
        since midnight, Oct 14, 1582 (the beginning of the Gregorian calendar))
        to a human-readable format"""
        try:
            if not hasattr(self, "gregorianEpoch"):
                self.gregorianEpoch = datetime.datetime(1582, 10, 14, 0, 0, 0)
            theDate = (self.gregorianEpoch +
                       datetime.timedelta(seconds=spssDateValue))
            return bytez(datetime.datetime.strftime(theDate, fmt))
        except (OverflowError, TypeError, ValueError):
            return recodeSysmisTo

    def getFileReport(self):
        """ This function prints a report about basic file characteristics """
        filesize = os.path.getsize(self.savFileName)
        kb = float(filesize) / 2**10
        mb = float(filesize) / 2**20
        (fileSize, label) = (mb, "MB") if mb > 1 else (kb, "kB")
        systemString = self.systemString.decode(self.fileEncoding)
        spssVersion = ".".join(map(str, self.spssVersion))
        lang, cp = locale.getlocale()
        intEnc = "Utf-8/Unicode" if self.ioUtf8 else "Codepage (%s)" % cp
        varlist = []
        line = "  %%0%sd. %%s (%%s - %%s)" % len(str(len(self.varNames) + 1))
        for cnt, varName in enumerate(self.varNames):
            lbl = "string" if self.varTypes[varName] > 0 else "numerical"
            format_ = self.formats[varName].decode(self.fileEncoding)
            varName = varName.decode(self.fileEncoding) 
            varlist.append(line % (cnt + 1, varName, format_, lbl))
        info = {"savFileName": self.savFileName,
                "fileSize": fileSize,
                "label": label,
                "nCases": self.nCases,
                "nCols": len(self.varNames),
                "nValues": self.nCases * len(self.varNames),
                "spssVersion": "%s (%s)" % (systemString, spssVersion),
                "ioLocale": self.ioLocale.decode(self.fileEncoding),
                "ioUtf8": intEnc,
                "fileEncoding": self.fileEncoding,
                "fileCodePage": self.fileCodePage,
                "isCompatible": "Yes" if self.isCompatibleEncoding() else "No",
                "local_language": lang,
                "local_encoding": cp,
                "varlist": os.linesep.join(varlist),
                "sep": os.linesep,
                "asterisks": 70 * "*"}
        report = ("%(asterisks)s%(sep)s" +
                  "*File '%(savFileName)s' (%(fileSize)3.2f %(label)s) has " +
                  "%(nCols)s columns (variables) and %(nCases)s rows " +
                  "(%(nValues)s values)%(sep)s" +
                  "*The file was created with SPSS version: %(spssVersion)s%" +
                  "(sep)s" +
                  "*The interface locale is: '%(ioLocale)s'%(sep)s" +
                  "*The interface mode is: %(ioUtf8)s%(sep)s" +
                  "*The file encoding is: '%(fileEncoding)s' (Code page: " +
                  "%(fileCodePage)s)%(sep)s" +
                  "*File encoding and the interface encoding are compatible:" +
                  " %(isCompatible)s%(sep)s" +
                  "*Your computer's locale is: '%(local_language)s' (Code " +
                  "page: %(local_encoding)s)%(sep)s" +
                  "*The file contains the following variables:%(sep)s" +
                  "%(varlist)s%(sep)s%(asterisks)s%(sep)s") % info
        if hasattr(report, "decode"):
            report = report.decode(self.fileEncoding)
        return report

    def getHeader(self, selectVars):
        """This function returns the variable names, or a selection thereof
        (as specified as a list using the selectVars parameter), as a list."""
        if selectVars is None:
            header = self.varNames
        elif isinstance(selectVars, (list, tuple)):
            diff = set(selectVars).difference(set(self.varNames))
            if diff:
                msg = "Variable names misspecified (%r)" % ", ".join(diff)
                raise NameError(msg)
            varPos = [self.varNames.index(v) for v in self.varNames
                      if v in selectVars]
            self.selector = operator.itemgetter(*varPos)
            header = self.selector(self.varNames)
            header = [header] if not isinstance(header, tuple) else header
        else:
            msg = ("Variable names list misspecified. Must be 'None' or a " +
                   "list or tuple of existing variables")
            raise TypeError(msg)
        return header
