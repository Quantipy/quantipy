.. savReaderWriter documentation master file, created by
   sphinx-quickstart on Thu Jan  3 00:25:18 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2

Welcome to savReaderWriter's documentation!
=================================================================================

.. module:: savReaderWriter
   :platform: Linux, Windows, Mac OS, HP-UX, AIX, Solaris, zLinux
   :synopsis: Read/Write SPSS system files (.sav, .zsav)
.. moduleauthor:: Albert-Jan Roskam

.. _`IBM SPSS Statistics Command Syntax Reference.pdf`: ftp://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Command_Syntax_Reference.pdf
.. _`International License Agreement`: ./_static/LA_en

In the documentation below, the associated SPSS commands are given in ``CAPS``.
See also the `IBM SPSS Statistics Command Syntax Reference.pdf`_ for info about SPSS syntax.

.. raw:: html

    <embed>
    </p>I always appreciate getting
    <script type="text/javascript" language="javascript">
    <!--
    // Email obfuscator script 2.1 by Tim Williams, University of Arizona
    // Random encryption key feature by Andrew Moulden, Site Engineering Ltd
    // This code is freeware provided these four comment lines remain intact
    // A wizard to generate this code is at http://www.jottings.com/obfuscator/
    { coded = "KNMT1@S8oNN.TNM"
      key = "NVXDIRH5nwJ1dLckfsjZFzbCv79xYTWKh3qytUuam0O4PpioEASG628MerlQgB"
      shift=coded.length
      link=""
      for (i=0; i<coded.length; i++) {
        if (key.indexOf(coded.charAt(i))==-1) {
          ltr = coded.charAt(i)
          link += (ltr)
        }
        else {     
          ltr = (key.indexOf(coded.charAt(i))-shift+key.length) % key.length
          link += (key.charAt(ltr))
        }
      }
    document.write("<a href='mailto:"+link+"?subject=feedback on savReaderWriter'>feedback</a>")
    }
    //-->
    </script><noscript>Sorry, you need Javascript on to email me.</noscript>
    on this package, so I can keep improving it!</p>
    </embed>

.. seealso::

   The :mod:`savReaderWriter` program uses the SPSS I/O module (``.so``, ``.dll``, ``.dylib``, depending on your Operating  System). Users of the SPSS I/O
   module should read the `International License Agreement`_ before using the SPSS I/O module. By downloading, installing, copying, accessing, or otherwise
   using the  SPSS I/O module, licensee agrees to the terms of this agreement. Copyright © IBM Corporation™ 1989, 2012 --- all rights reserved.

Installation
============================================================================

Platforms
----------
As shown in **Table 0** below, this program works for Linux (incl. z/Linux), Windows, Mac OS (32 and 64 bit), AIX-64, HP-UX and Solaris-64. However, it has only been tested on Linux 32 (Ubuntu and Mint), Windows (mostly on Windows XP 32, but also a few times on Windows 7 64), and Mac OS (with an earlier version of savReaderWriter). The other OSs are entirely untested.

.. exceltable:: **Table 0.** supported platforms for ``savReaderWriter`` 
   :file: ./platforms.xls
   :header: 2
   :selection: A1:C9

Setup
-------------------
The program can be installed by running::

    python setup.py install

Or alternatively::

    pip install savReaderWriter --allow-all-external

To get the 'bleeding edge' version straight from the repository do::

    pip install -U -e git+https://bitbucket.org/fomcl/savreaderwriter.git#egg=savreaderwriter

.. versionchanged:: 3.3
* The ``savReaderWriter`` program now runs on Python 2 and 3. It is tested with Python 2.7, 3.3 and PyPy
* Several bugs were removed, notably two that prevented the I/O modules from loading in 64-bit Linux and 64-bit Windows systems (NB: these bugs were entirely unrelated). In addition, long variable labels were truncated to 120 characters, which is now fixed.

.. versionchanged:: 3.2

* The ``savReaderWriter`` program is now self-contained. That is, the IBM SPSS I/O modules now all load by themselves, without any changes being required anymore to ``PATH``, ``LD_LIBRARY_PATH`` and equivalents. Also, no extra .deb files need to be installed anymore (i.e. no dependencies). 

* ``savReaderWriter`` now uses version 21.0.0.1 (i.e., Fixpack 1) of the I/O module.

Optional features
-------------------

**cWriterow.**
The ``cWriterow`` package is a faster Cython implementation of the pyWriterow method (66 % faster). To install it, you need Cython and run ``setup.py`` in the ``cWriterow`` folder::

    easy_install cython
    python setup.py build_ext --inplace

**psyco.**
The ``psyco`` package may be installed to speed up reading (66 % faster).

**numpy.**
The ``numpy`` package should be installed if you intend to use array slicing (e.g ``data[:2,2:4]``).

:mod:`SavWriter` -- Write SPSS system files
============================================================================
.. function:: SavWriter(savFileName, varNames, varTypes, [valueLabels=None, varLabels=None, formats=None, missingValues=None, measureLevels=None, columnWidths=None, alignments=None, varSets=None, varRoles=None, varAttributes=None, fileAttributes=None, fileLabel=None, multRespDefs=None, caseWeightVar=None, overwrite=True, ioUtf8=False, ioLocale=None, mode="wb", refSavFileName=None])
   
   **Write SPSS system files (.sav, .zsav)**

   :param savFileName: the file name of the spss data file. File names that end with '.zsav' are   compressed using the ZLIB (ZSAV) compression scheme (requires v21 SPSS I/O files), while for file    names that end with '.sav' the 'old' compression scheme is used (it is not possible to generate uncompressed files unless you modify the source code).

   :param varNames: list of the variable names in the order in which they appear in the spss data file.

   :param varTypes: varTypes dictionary ``{varName: varType}``, where varType == 0 means 'numeric', and varType > 0 means 'character' of that length (in bytes)

   :param valueLabels: value label dictionary ``{varName: {value: label}}``. Cf. ``VALUE LABELS`` (default: ``None``).

   :param varLabels: variable label dictionary ``{varName: varLabel}``. Cf. ``VARIABLE LABEL`` (default: ``None``).

   :param formats: print/write format dictionary ``{varName: spssFmt}``. Commonly used formats include F  (numeric, e.g. ``F5.4``), N (numeric with leading zeroes, e.g. ``N8``), A (string, e.g. ``A8``) and ``EDATE``/``ADATE`` (European/American date, e.g. ``ADATE30``). Cf. ``FORMATS`` (default: ``None``). See also under `Formats`_.

   :param missingValues: missing values dictionary ``{varName: {missing_value_spec}}``. Cf. ``MISSING VALUES`` (default: ``None``). For example: 

      .. code:: python

         missingValues = {"someNumvar1": {"values": [999, -1, -2]},  # discrete values
                          "someNumvar2": {"lower": -9, "upper": -1}, # range, cf. MISSING VALUES x (-9 THRU -1)
                          "someNumvar3": {"lower": -9, "upper": -1, "value": 999},
                          "someStrvar1": {"values": ["foo", "bar", "baz"]},
                          "someStrvar2": {"values': "bletch"}}
     
      .. warning:: *measureLevels, columnWidths, alignments must all three be set, if used*

   :param measureLevels: measurement level dictionary ``{varName: <level>}``. Valid levels are: "unknown",  "nominal", "ordinal", "scale", "ratio", "flag", "typeless". Cf. ``VARIABLE LEVEL`` (default: ``None``). 

   :param columnWidths: column display width dictionary ``{varName: <int>}``. Cf. ``VARIABLE WIDTH``.   (default: ``None`` --> >= 10 [stringVars] or automatic [numVars]). 

   :param alignments: alignment dictionary ``{varName: <left/center/right>}`` Cf. ``VARIABLE ALIGNMENT``  (default: ``None`` --> numerical: right, string: left). 

   :param varSets: sets dictionary ``{setName: [list_of_valid_varNames]}``. Cf. ``SETSMACRO`` extension  command. (default: ``None``). 

   :param varRoles: variable roles dictionary ``{varName: varRole}``. VarRoles may be any of the following:  'both', 'frequency', 'input', 'none', 'partition', 'record ID', 'split', 'target'. Cf. ``VARIABLE ROLE``  (default: ``None``). 

   :param varAttributes: variable attributes dictionary ``{varName: {attribName: attribValue}`` Cf. ``VARIABLE  ATTRIBUTES``. (default: ``None``). For example:

      .. code:: python

         varAttributes = {'gender': {'Binary': 'Yes'}, 'educ': {'DemographicVars': '1'}}

   :param fileAttributes: file attributes dictionary ``{attribName: attribValue}``. Square brackets indicate  attribute arrays, which must  start with 1. Cf. ``FILE ATTRIBUTES``. (default: ``None``). For example:

      .. code:: python

         fileAttributes = {'RevisionDate[1]': '10/29/2004', 'RevisionDate[2]': '10/21/2005'} 

   :param fileLabel: file label string, which defaults to "File created by user <username> at <datetime>" if file label is ``None``. Cf. ``FILE LABEL`` (default: ``None``). 

   :param multRespDefs: Multiple response sets definitions (dichotomy groups or category groups) dictionary ``{setName: <set definition>}``. In SPSS syntax, 'setName' has a dollar prefix ('$someSet'). See also  docstring of multRespDefs method. Cf. ``MRSETS``. (default: ``None``). 

   :param caseWeightVar: valid varName that is set as case weight. Cf. ``WEIGHT BY`` command. 

   :param overwrite: Boolean that indicates whether an existing SPSS file should be overwritten (default: ``True``). 

   :param ioUtf8: Boolean that indicates the mode in which text communicated to or from the I/O Module will  be. Valid values are ``True`` (UTF-8/unicode mode, cf. ``SET UNICODE=ON``) or ``False`` (Codepage mode, ``SET  UNICODE=OFF``) (default: ``False``). 

   :param ioLocale: indicates the locale of the I/O module, cf. ``SET LOCALE`` (default: ``None``, which is the  same as ``".".join(locale.getlocale())``. Locale specification is OS-dependent. See also under ``SavHeaderReader``.

   :param mode: indicates the mode in which <savFileName> should be opened. Possible values are "wb" (write),  "ab" (append), "cp" (copy: initialize header using <refSavFileName> as a reference file, cf. ``APPLY DICTIONARY``). (default: "wb"). 

   :param refSavFileName: reference file that should be used to initialize the header (aka the SPSS data  dictionary) containing variable label, value label, missing value, etc, etc definitions. Only relevant  in conjunction with mode="cp". (default: ``None``). 


Typical use::
    
    savFileName = 'someFile.sav'
    records = [['Test1', 1, 1], ['Test2', 2, 1]]
    varNames = ['var1', 'v2', 'v3']
    varTypes = {'var1': 5, 'v2': 0, 'v3': 0}
    with SavWriter(savFileName, varNames, varTypes) as writer:
        for record in records:
            writer.writerow(record)

.. seealso::

    More code examples can be found in the ``doc_tests`` folder

:mod:`SavReader` -- Read SPSS system files
============================================================================
.. function:: SavReader(savFileName, [returnHeader=False, recodeSysmisTo=None, verbose=False, selectVars=None, idVar=None, rawMode=False, ioUtf8=False, ioLocale=None])

   **Read SPSS system files (.sav, .zsav)**

   :param savFileName: the file name of the spss data file

   :param returnHeader: Boolean that indicates whether the first record should be a list of variable names (default = ``False``)

   :param recodeSysmisTo: indicates to which value missing values should be recoded (default = ``None``, i.e. no recoding is done)

   :param selectVars: indicates which variables in the file should be selected. The variables should be  specified as a list or a tuple of valid variable names. If None is specified, all variables in the file are used (default = ``None``)

   :param idVar: indicates which variable in the file should be used for use as id variable for the 'get'  method (default = ``None``)

   :param verbose: Boolean that indicates whether information about the spss data file (e.g., number of cases,  variable names, file size) should be printed on the screen (default = ``False``).

   :param rawMode: Boolean that indicates whether values should get SPSS-style formatting, and whether date variables (if present) should be converted to ISO-dates. If True, the program does not format any values, which increases processing speed. (default = ``False``)

   :param ioUtf8: Boolean that indicates the mode in which text communicated to or from the I/O Module will be. Valid values are True (UTF-8 mode aka Unicode mode) and False (Codepage mode). Cf. ``SET UNICODE=ON/OFF`` (default = ``False``)

   :param ioLocale: indicates the locale of the I/O module. Cf. ``SET LOCALE`` (default = ``None``, which corresponds to ``".".join(locale.getlocale())``). See also under ``SavHeaderReader``.

.. warning::

   Once a file is open, ``ioUtf8`` and ``ioLocale`` can not be changed. The same applies after a file could not be successfully closed. Always ensure a file is closed by calling ``__exit__()`` (i.e., using a context manager) or ``close()`` (in a ``try - finally`` suite)

Typical use::
    
    savFileName = "someFile.sav"
    with SavReader(savFileName, returnHeader=True) as reader:
        header = next(reader)
        for line in reader:
            process(line)

Use of ``__getitem__`` and other methods::
    
    data = SavReader(savFileName, idVar="id")
    with data:
        print("The file contains %d records" % len(data))
        print(str(data))  # prints a file report
        print("The first six records look like this\n"), data[:6]
        print("The first record looks like this\n"), data[0]
        print("The last four records look like this\n"), data.tail(4)
        print("The first five records look like this\n"), data.head()
	allData = data.all()
        print("First column:\n"), data[..., 0]  # requires numpy
        print("Row 4 & 5, first three cols\n"), data[4:6, :3]  # requires numpy
        ## ... Do a binary search for records --> idVar
        print(data.get(4, "not found"))  # gets 1st record where id==4


.. seealso::

    More code examples can be found in the ``doc_tests`` folder

:mod:`SavHeaderReader` -- Read SPSS file meta data
============================================================================
.. function:: SavHeaderReader(savFileName[, ioUtf8=False, ioLocale=None])
	
   **Read SPSS file meta data. Yields the same information as the SPSS command ``DISPLAY DICTIONARY``**


   :param savFileName: the file name of the spss data file

   :param ioUtf8: Boolean that indicates the mode in which text communicated to or from the I/O Module will be. Valid values are ``True`` (UTF-8 mode aka Unicode mode) and ``False`` (Codepage mode). Cf. ``SET UNICODE=ON/OFF`` (default = ``False``)

   :param ioLocale: indicates the locale of the I/O module. Cf. ``SET LOCALE`` (default = ``None``, which corresponds to ``".".join(locale.getlocale())``). Example where this may be needed::

      .. code:: python

         # wrong:  variables are returned as v1, v2, v3
         >>> with SavHeaderReader('german.sav') as header:
         ...     print(header.varNames)
         [b'python', b'programmieren', b'macht', b'v1', b'v2', b'v3']

         # correct: variable names contain non-ascii characters
         # locale definition and presence is OS-specific
         # Linux: sudo localedef -f CP1252 -i de_DE /usr/lib/locale/de_DE.cp1252
         >>> with SavHeaderReader('german.sav', ioLocale='de_DE.cp1252') as header:
         ...     print(header.varNames)
         [b'python', b'programmieren', b'macht', b'\xfcberhaupt', b'v\xf6llig', b'spa\xdf']

.. warning::

   The program calls ``spssFree*`` C functions to free memory allocated to dynamic arrays. This previously sometimes caused segmentation faults. This problem now appears to be solved. However, if you do experience segmentation faults you can set ``segfaults=True`` in ``__init__.py``. This will prevent the spssFree* functions from being called (and introduce a memory leak).

Typical use::

    with SavHeaderReader(savFileName) as header:
        metadata = header.dataDictionary()
        report = str(header)
        print(report)

.. seealso::

    More code examples can be found in the ``doc_tests`` folder

Formats
----------

SPSS knows just two different data types: string and numerical data. These data types can be *formatted* (displayed) by SPSS in several different ways. Format names are followed by total width (w) and an optional number of decimal positions (d). **Table 1** below shows a complete list of all the available formats.

**String** data can be alphanumeric characters (``A`` format) or the hexadecimal representation of alphanumeric characters (``AHEX`` format). The maximum size of a string value is 32767 bytes. String formats do not have any decimal positions (d). Currently, ``SavReader`` maps both of the string formats to a regular alphanumeric string format. 

**Numerical** data formats include the default numeric format (``F``), scientific notation (``E``) and zero-padded (``N``). For example, a format of ``F5.2`` represents a numeric value with a total width of 5, including two decimal positions and a decimal indicator. For all numeric formats, the maximum width (w) is 40. For numeric formats where decimals are allowed, the maximum number of decimals (d) is 16. ``SavReader`` does not format numerical values, except for the ``N`` format, and dates/times (see under `Date formats`_). The ``N`` format is a zero-padded value (e.g. SPSS format ``N8`` is formatted as Python format ``%08d``, e.g. '00001234'). For most numerical values, formatting means *loss of precision*. For instance, formatting SPSS ``F5.3`` to Python ``%5.3f`` means that only the first three digits are retained. In addition, formatting incurs *additional processing time*. Finally, e.g. appending a percent sign to a value (``PCT`` format) renders the value *less useful for calculations*.

.. exceltable:: **Table 1.** string and numerical formats in SPSS and ``savReaderWriter`` 
   :file: ./formats.xls
   :header: 1
   :selection: A1:D19
*Note.* The User Programmable currency formats (CCA, CCB, CCC and CCD) cannot be defined or written by ``SavWriter`` and existing definitions cannot be read by ``SavReader``.

Date formats
-------------
**Dates in SPSS.** Date formats are a group of numerical formats. SPSS stores dates as the number of seconds since midnight, October 14, 1582 (the beginning of the Gregorian calendar). In SPSS, the user can make these seconds human-readable by giving them a print and/or write format (usually these are set at the same time using the ``FORMATS`` command). Examples of such display formats include ``ADATE`` (American date, *mmddyyyy*) and ``EDATE`` (European date, *ddmmyyyy*), ``SDATE`` (Asian/Sortable date, *yyyymmdd*) and ``JDATE`` (Julian date). 

**Reading dates.** ``SavReader`` deliberately does *not* honor the different SPSS date display formats, but instead tries to convert them to the more practical (sortable) and less ambibiguous ISO 8601 format (*yyyy-mm-dd*). You can easily change this behavior by modifying the ``supportedDates`` dictionary in ``__init__.py``. **Table 2** below shows how ``SavReader`` converts SPSS dates. Where applicable, the SPSS-to-Python conversion always results in the 'long' version of a date/time. For instance, ``TIME5`` and ``TIME40.16`` both result in a ``%H:%M:%S.%f``-style format. If you do not want ``SavReader`` to automatically convert dates, you can set ``rawMode=True``. If you use this setting, keep in mind that ``SavReader`` will also not convert system missing values (``$SYSMIS``) to an empty string; instead sysmis values will appear as the smallest value that can be represented on that system (``-1 * sys.float_info.max``)

.. exceltable:: **Table 2.** Date formats in SPSS and ``SavReader`` 
   :file: ./dates.xls
   :header: 1
   :selection: A1:I25
*Note.*
[1] `IBM SPSS Statistics Command Syntax Reference.pdf`_
[2] http://docs.python.org/2/library/datetime.html
[3] ISO 8601 format dates are used wherever possible, e.g. mmddyyyy (``ADATE``) and ddmmyyyy (``EDATE``) is not maintained.
[4] Months are converted to quarters using a simple lookup table
[5] weekday, month names depend on host locale (not on ``ioLocale`` argument)

**Writing dates.** With ``SavWriter`` a Python date string value (e.g. "2010-10-25") can be converted to an SPSS Gregorian date (i.e., just a whole bunch of seconds) by using the ``spssDateTime`` method, e.g.::

    kwargs = dict(savFileName='/tmp/date.sav', varNames=['aDate'], varTypes={'aDate': 0}, formats={'aDate': 'EDATE40'})
    with SavWriter(**kwargs) as writer:
        spssDateValue = writer.spssDateTime('2010-10-25', '%Y-%m-%d')
        writer.writerow([spssDateValue])

The display format of the date (i.e., the way it looks in the SPSS data editor after opening the .sav file) may be set by specifying the ``formats`` dictionary (see also **Table 1**). This is one of the optional arguments of the ``SavWriter`` initializer. Without such a specification, the date will look like a large integer (the number of seconds since the beginning of the Gregorian calendar).



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

