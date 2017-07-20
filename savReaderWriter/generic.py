#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ctypes import *
import ctypes.util
import struct
import sys
import platform
import os
import re
import math
import locale
import encodings
import collections

from savReaderWriter import *
from py3k import *

class Generic(object):
    """
    Class for methods and data used in reading as well as writing
    IBM SPSS Statistics data files
    """

    def __init__(self, savFileName, ioUtf8=False, ioLocale=None):
        """Constructor. Note that interface locale and encoding can only
        be set once"""
        locale.setlocale(locale.LC_ALL, "")
        self.savFileName = savFileName
        self.libc = cdll.LoadLibrary(ctypes.util.find_library("c"))
        self.spssio = self.loadLibrary()

        self.wholeCaseIn = self.spssio.spssWholeCaseIn
        self.wholeCaseOut = self.spssio.spssWholeCaseOut

        self.encoding_and_locale_set = False
        if not self.encoding_and_locale_set:
            self.encoding_and_locale_set = True
            self.ioLocale = ioLocale
            self.ioUtf8 = ioUtf8

    def _encodeFileName(self, fn):
        """Helper function to encode unicode file names into bytestring file
        names encoded in the file system's encoding. Needed for C functions
        that have a c_char_p filename argument.
        http://effbot.org/pyref/sys.getfilesystemencoding.htm
        http://docs.python.org/2/howto/unicode.html under 'unicode filenames'"""
        if not isinstance(fn, unicode):
            return fn
        elif sys.platform.startswith("win"):
            return self.wide2utf8(fn)
        else:
            encoding = sys.getfilesystemencoding()
            encoding = "utf-8" if not encoding else encoding  # actually, ascii
        try:
            return fn.encode(encoding)
        except UnicodeEncodeError:
            msg = ("File system encoding %r can not be used to " +
                   "encode file name %r [%s]")
            raise ValueError(msg % (encoding, fn, sys.exc_info()[1]))

    def _loadLibs(self, folder):
        """Helper function that loads I/O libraries in the correct order"""
        # Get a list of all the files in the spssio dir for a given OS
        # Sort the list in the order in which the libs need to be loaded
        # Using regex patterns ought to be more resilient to updates of the
        # I/O modules, compared to hardcoding the names
        debug = False
        path = os.path.join(os.path.dirname(__file__), "spssio", folder)
        libs = sorted(os.listdir(path))

        pats = ['(lib)?icuda?t', '(lib)?icuuc', '(lib)?icui',
                '(lib)?zlib', '(lib)?spssd?io']
        libs = [lib for pat in pats for lib in libs if re.match(pat, lib)]
        isLib = r"""\w+(\.s[ol](?:\.\d+)*| # linux/hp/solaris
                    \.\d+\.a|              # aix
                    \.dll|                 # windows
                    (\.\d+)*\.dylib)$      # mac""" # filter out non-libs
        libs = [lib for lib in libs if re.match(isLib, lib, re.I | re.X)]
        load = WinDLL if sys.platform.lower().startswith("win") else CDLL
        if libs and debug:
            print(os.path.basename(path).upper().center(79, "-"))
            print("\n".join(libs))

        return [load(os.path.join(path, lib)) for lib in libs][-1]

    def loadLibrary(self):
        """This function loads and returns the SPSSIO libraries,
        depending on the platform."""

        arch = platform.architecture()[0]
        is_32bit, is_64bit = arch == "32bit", arch == "64bit"
        pf = sys.platform.lower()

        # windows
        if pf.startswith("win") and is_32bit:
            spssio = self._loadLibs("win32")
        elif pf.startswith("win"):
            spssio = self._loadLibs("win64")

        # linux
        elif pf.startswith("lin") and is_32bit:
            spssio = self._loadLibs("lin32")
        elif pf.startswith("lin") and is_64bit and os.uname()[-1] == "s390x":
            # zLinux64: Thanks Anderson P. from System z Linux LinkedIn Group!
            spssio = self._loadLibs("zlinux")
        elif pf.startswith("lin") and is_64bit:
            spssio = self._loadLibs("lin64")

        # other
        elif pf.startswith("darwin") or pf.startswith("mac"):
            # Mac: Thanks Rich Sadowsky!
            spssio = self._loadLibs("macos")
        elif pf.startswith("aix") and is_64bit:
            spssio = self._loadLibs("aix64")
        elif pf.startswith("hp-ux"):
            spssio = self._loadLibs("hpux_it")
        elif pf.startswith("sunos") and is_64bit:
            spssio = self._loadLibs("sol64")
        else:
            msg = "Your platform (%r) is not supported" % pf
            raise EnvironmentError(msg)

        return spssio

    def errcheck(self, res, func, args):
        """This function checks for errors during the execution of
        function <func>"""
        if not res:
            msg = "Error performing %r operation on file %r."
            raise IOError(msg % (func.__name__, self.savFileName))
        return res

    def wide2utf8(self, fn):
        """Take a unicode file name string and encode it to a multibyte string
        that Windows can use to represent file names (CP65001, UTF-8)
        http://msdn.microsoft.com/en-us/library/windows/desktop/dd374130"""

        from ctypes import wintypes

        _CP_UTF8 = 65001
        _CP_ACP = 0  # ANSI
        _LPBOOL = POINTER(c_long)

        _wideCharToMultiByte = windll.kernel32.WideCharToMultiByte
        _wideCharToMultiByte.restype = c_int
        _wideCharToMultiByte.argtypes = [wintypes.UINT, wintypes.DWORD,
                                         wintypes.LPCWSTR, c_int,
                                         wintypes.LPSTR, c_int,
                                         wintypes.LPCSTR, _LPBOOL]
        codePage = _CP_UTF8
        dwFlags = 0
        lpWideCharStr = fn
        cchWideChar = len(fn)
        lpMultiByteStr = None
        cbMultiByte = 0  # zero requests size
        lpDefaultChar = None
        lpUsedDefaultChar = None

        # get size
        mbcssize = _wideCharToMultiByte(
        codePage, dwFlags, lpWideCharStr, cchWideChar, lpMultiByteStr,
        cbMultiByte, lpDefaultChar, lpUsedDefaultChar)
        if mbcssize <= 0:
            raise WinError(mbcssize)
        lpMultiByteStr = create_string_buffer(mbcssize)

        # convert
        retcode = _wideCharToMultiByte(
        codePage, dwFlags, lpWideCharStr, cchWideChar, lpMultiByteStr,
        mbcssize, lpDefaultChar, lpUsedDefaultChar)
        if retcode <= 0:
            raise WinError(retcode)
        return lpMultiByteStr.value

    def openSavFile(self, savFileName, mode=b"rb", refSavFileName=None):
        """This function opens IBM SPSS Statistics data files in mode <mode>
        and returns a handle that  should be used for subsequent operations on
        the file. If <savFileName> is opened in mode "cp", meta data
        information aka the spss dictionary is copied from <refSavFileName>"""
        savFileName = os.path.abspath(savFileName)  # fdopen wants full name
        try:
            fdopen = self.libc._fdopen  # Windows
        except AttributeError:
            fdopen = self.libc.fdopen   # Linux and others
        fdopen.argtypes, fdopen.restype = [c_int, c_char_p], c_void_p
        fdopen.errcheck = self.errcheck
        mode_ = b"wb" if mode == b"cp" else mode
        with open(savFileName, mode_.decode("utf-8")) as f:
            self.fd = fdopen(f.fileno(), mode_)
        if mode == b"rb":
            spssOpen = self.spssio.spssOpenRead
        elif mode == b"wb":
            spssOpen = self.spssio.spssOpenWrite
        elif mode == b"cp":
            spssOpen = self.spssio.spssOpenWriteCopy
        elif mode == b"ab":
            spssOpen = self.spssio.spssOpenAppend

        savFileName = self._encodeFileName(savFileName)
        refSavFileName = self._encodeFileName(refSavFileName)
        sav = c_char_py3k(savFileName)
        fh = c_int(self.fd)
        if mode == b"cp":
            retcode = spssOpen(sav, c_char_py3k(refSavFileName), pointer(fh))
        else:
            retcode = spssOpen(sav, pointer(fh))

        msg = "Problem opening file %r in mode %r" % (savFileName, mode)
        checkErrsWarns(msg, retcode)
        return fh.value

    def closeSavFile(self, fh, mode=b"rb"):
        """This function closes the sav file associated with <fh> that was open
        in mode <mode>."""
        if mode == b"rb":
            spssClose = self.spssio.spssCloseRead
        elif mode in (b"wb", b"cp"):
            spssClose = self.spssio.spssCloseWrite
        elif mode == b"ab":
            spssClose = self.spssio.spssCloseAppend
        else:
            spssClose, retcode = None, 9999

        if spssClose is not None:
            retcode = spssClose(c_int(fh))
        msg = "Problem closing file in mode %r" % mode
        checkErrsWarns(msg, retcode)
        #self.libc.close(self.fd)  ???

    @property
    def releaseInfo(self):
        """This function reports release- and machine-specific information
        about the open file."""
        relInfo = ["release number", "release subnumber", "fixpack number",
                   "machine code", "floating-point representation code",
                   "compression scheme code", "big/little-endian code",
                   "character representation code"]
        relInfoArr = (c_int * len(relInfo))()
        retcode = self.spssio.spssGetReleaseInfo(c_int(self.fh), relInfoArr)
        checkErrsWarns("Problem getting ReleaseInfo", retcode)
        info = dict([(item, relInfoArr[i]) for i, item in enumerate(relInfo)])
        return info

    @property
    def spssVersion(self):
        """Return the SPSS version that was used to create the opened file
        as a three-tuple indicating major, minor, and fixpack version as
        ints. NB: in the transition from SPSS to IBM, a new four-digit
        versioning nomenclature is used. This function returns the old
        three-digit nomenclature. Therefore, no patch version information
        is available."""
        info = self.releaseInfo
        major = info["release number"]
        minor = info["release subnumber"]
        fixpack = info["fixpack number"]
        ver_info = (major, minor, fixpack)
        return collections.namedtuple("ver", "major minor fixpack")(*ver_info)

    @property
    def fileCompression(self):
        """Get/Set the file compression.
        Returns/Takes a compression switch which may be any of the following:
        'uncompressed', 'standard', or 'zlib'. Zlib comression requires SPSS
        v21 I/O files."""
        compression = {0: b"uncompressed", 1: b"standard", 2: b"zlib"}
        compSwitch = c_int()
        func = self.spssio.spssGetCompression
        retcode = func(c_int(self.fh), byref(compSwitch))
        checkErrsWarns("Problem getting file compression", retcode)
        return compression.get(compSwitch.value)

    @fileCompression.setter
    def fileCompression(self, compSwitch):
        compression = {b"uncompressed": 0, b"standard": 1, b"zlib": 2}
        compSwitch = compression.get(compSwitch)
        func = self.spssio.spssSetCompression
        retcode = func(c_int(self.fh), c_int(compSwitch))
        invalidSwitch = retcodes.get(retcode) == 'SPSS_INVALID_COMPSW'
        if invalidSwitch and self.spssVersion[0] < 21:
            msg = "Writing zcompressed files requires >=v21 SPSS I/O libraries"
            raise ValueError(msg)
        checkErrsWarns("Problem setting file compression", retcode)

    @property
    def systemString(self):
        """This function returns the name of the system under which the file
        was created aa a string."""
        sysName = create_string_buffer(42)
        func = self.spssio.spssGetSystemString
        retcode = func(c_int(self.fh), byref(sysName))
        checkErrsWarns("Problem getting SystemString", retcode)
        return sysName.value

    def getStruct(self, varTypes, varNames, mode=b"rb"):
        """This function returns a compiled struct object. The required
        struct format string for the conversion between C and Python
        is created on the basis of varType and byte order.
        --varTypes: SPSS data files have either 8-byte doubles/floats or n-byte
          chars[]/ strings, where n is always 8 bytes or a multiple thereof.
        --byte order: files are written in the byte order of the host system
          (mode="wb") and read/appended using the byte order information
          contained in the SPSS data file (mode is "ab" or "rb" or "cp")"""
        if mode in (b"ab", b"rb", b"cp"):   # derive endianness from file
            endianness = self.releaseInfo["big/little-endian code"]
            endianness = ">" if endianness > 0 else "<"
        elif mode == b"wb":                 # derive endianness from host
            if sys.byteorder == "little":
                endianness = "<"
            elif sys.byteorder == "big":
                endianness = ">"
            else:
                endianness = "@"
        structFmt = [endianness]
        ceil = math.ceil
        for varName in varNames:
            varType = varTypes[varName]
            if varType == 0:
                structFmt.append("d")
            else:
                fmt = str(int(ceil(int(varType) / 8.0) * 8))
                structFmt.append(fmt + "s")
        return struct.Struct("".join(structFmt))

    def getCaseBuffer(self):
        """This function returns a buffer and a pointer to that buffer. A whole
        case will be read into this buffer."""
        caseSize = c_long()
        retcode = self.spssio.spssGetCaseSize(c_int(self.fh), byref(caseSize))
        caseBuffer = create_string_buffer(caseSize.value)
        checkErrsWarns("Problem getting case buffer", retcode)
        return caseBuffer

    @property
    def sysmis(self):
        """This function returns the IBM SPSS Statistics system-missing
        value ($SYSMIS) for the host system (also called 'NA' in other
        systems)."""
        try:
            sysmis = -1 * sys.float_info[0]  # Python 2.6 and higher.
        except AttributeError:
            self.spssio.spssSysmisVal.restype = c_float
            sysmis = self.spssio.spssSysmisVal()
        return sysmis

    @property
    def missingValuesLowHigh(self):
        """This function returns the 'lowest' and 'highest' values used for
        numeric missing value ranges on the host system. This can be used in
        a similar way as the LO and HI keywords in missing values
        specifications (cf. MISSING VALUES foo (LO THRU 0). It may be called
        at any time."""
        lowest, highest = c_double(), c_double()
        func = self.spssio.spssLowHighVal
        retcode = func(byref(lowest), byref(highest))
        checkErrsWarns("Problem getting min/max missing values", retcode)
        ranges = (lowest.value, highest.value)
        return collections.namedtuple("range", "lo hi")(*ranges)

    @property
    def ioLocale(self):
        """This function gets/sets the I/O Module's locale.
        This corresponds with the SPSS command SET LOCALE. The I/O Module's
        locale is separate from that of the client application. The
        <localeName> parameter and the return value are identical to those
        for the C run-time function setlocale. The exact locale name
        specification depends on the OS of the host sytem, but has the
        following form:
                   <lang>_<territory>.<codeset>[@<modifiers>]
        The 'codeset' and 'modifier' components are optional and in Windows,
        aliases (e.g. 'english') may be used. When the I/O Module is first
        loaded, its locale is set to the system default. See also:
        --https://wiki.archlinux.org/index.php/Locale
        --http://msdn.microsoft.com/en-us/library/39cwe7zf(v=vs.80).aspx"""
        if hasattr(self, "setLocale"):
            return self.setLocale
        else:
            currLocale = ".".join(locale.getlocale())
            print("NOTE. Locale not set; getting current locale: %s" % currLocale)
            return currLocale

    @ioLocale.setter
    def ioLocale(self, localeName=""):
        if not localeName:
            localeName = ".".join(locale.getlocale())
        func = self.spssio.spssSetLocale
        func.restype = c_char_p
        self.setLocale = func(c_int(locale.LC_ALL), c_char_py3k(localeName))
        if self.setLocale is None:
            raise ValueError("Invalid ioLocale: %r" % localeName)
        return self.setLocale

    @property
    def fileCodePage(self):
        """This function provides the Windows code page number of the encoding
        applicable to a file."""
        nCodePage = c_int()
        func = self.spssio.spssGetFileCodePage
        retcode = func(c_int(self.fh), byref(nCodePage))
        checkErrsWarns("Problem getting file codepage", retcode)
        return nCodePage.value

    def isCompatibleEncoding(self):
        """This function determines whether the file and interface encoding
        are compatible."""
        try:
            # Windows, note typo 'Endoding'!
            func = self.spssio.spssIsCompatibleEndoding
        except AttributeError:
            func = self.spssio.spssIsCompatibleEncoding
        func.restype = c_bool
        isCompatible = c_int()
        retcode = func(c_int(self.fh), byref(isCompatible))
        msg = "Error testing encoding compatibility: %r" % isCompatible.value
        checkErrsWarns(msg, retcode)
        if not isCompatible.value and not self.ioUtf8:
            msg = ("NOTE. SPSS Statistics data file %r is written in a " +
                   "character encoding (%s) incompatible with the current " +
                   "ioLocale setting. It may not be readable. Consider " +
                   "changing ioLocale or setting ioUtf8=True.")
            print(msg % (self.savFileName, self.fileEncoding))
        return bool(isCompatible.value)

    @property
    def ioUtf8(self):
        """This function returns/sets the current interface encoding.
        ioUtf8 = False --> CODEPAGE mode,
        ioUtf8 = True --> UTF-8 mode, aka. Unicode mode
        This corresponds with the SPSS command SHOW UNICODE (getter)
        and SET UNICODE=ON/OFF (setter)."""
        if hasattr(self, "ioUtf8_"):
            return self.ioUtf8_
        self.ioUtf8_ = self.spssio.spssGetInterfaceEncoding()
        return bool(self.ioUtf8_)

    @ioUtf8.setter
    def ioUtf8(self, ioUtf8):
        try:
            retcode = self.spssio.spssSetInterfaceEncoding(c_int(int(ioUtf8)))
            if retcode > 0 and not self.encoding_and_locale_set:
                # not self.encoding_and_locale_set --> nested context managers
                raise SPSSIOError("Error setting IO interface", retcode)
        except TypeError:
            msg = "Invalid interface encoding: %r (must be bool)" % ioUtf8
            raise Exception(msg)
        if retcode < 0:
            checkErrsWarns("Problem setting ioUtf8", retcode)

    @property
    def fileEncoding(self):
        """This function obtains the encoding applicable to a file.
        The encoding is returned as an IANA encoding name, such as
        ISO-8859-1, which is then converted to the corresponding Python
        codec name. If the file contains no file encoding, the locale's
        preferred encoding is returned"""
        try:
            pszEncoding = create_string_buffer(20)  # is 20 enough??
            func = self.spssio.spssGetFileEncoding
            retcode = func(c_int(self.fh), byref(pszEncoding))
            checkErrsWarns("Problem getting file encoding", retcode)
            iana_codes = encodings.aliases.aliases
            rawEncoding = pszEncoding.value.lower().decode("utf-8")
            if rawEncoding.replace("-", "") in iana_codes:
                iana_code = rawEncoding.replace("-", "")
            else:
                iana_code = rawEncoding.replace("-", "_")
            fileEncoding = iana_codes[iana_code]
            return fileEncoding
        except KeyError:
            print ("NOTE. IANA coding lookup error. Code %r " % iana_code +
                   "does not map to any Python codec.")
            return locale.getpreferredencoding()

    @property
    def record(self):
        """Get/Set a whole record from/to a pre-allocated buffer"""
        args = c_int(self.fh), byref(self.caseBuffer)
        retcode = self.wholeCaseIn(*args)
        if retcode:
            checkErrsWarns("Problem reading row", retcode)
        record = list(self.unpack_from(self.caseBuffer))
        return record

    @record.setter
    def record(self, record):
        try:
            self.pack_into(self.caseBuffer, 0, *record)
        except struct.error:
            msg = "Use ioUtf8=True to write unicode strings [%s]"
            raise TypeError(msg % sys.exc_info()[1])
        args = c_int(self.fh), c_char_py3k(self.caseBuffer.raw)
        retcode = self.wholeCaseOut(*args)
        if retcode:
            checkErrsWarns("Problem writing row\n" + record, retcode)

    def printPctProgress(self, nominator, denominator):
        """This function prints the % progress when reading and writing
        files"""
        if nominator and nominator % 10**4 == 0:
            pctProgress = (float(nominator) / denominator) * 100
            print("%2.1f%%... " % pctProgress),
