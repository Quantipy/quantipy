' ***************************************************************
' IBM Confidential
' 
' OCO Source Materials
' 
' IBM SPSS Products: Statistics Common
' 
' (C) Copyright IBM Corp. 1989, 2011
' 
' The source code for this program is not published or otherwise divested of its trade secrets, 
' irrespective of what has been deposited with the U.S. Copyright Office.
' ***************************************************************

Module SpssDioInterface

    ' Modification history:
    ' 16 Sep 96 - SPSS 7.5
    ' 05 Dec 97 - SPSS 8.0
    ' 27 Aug 98 - SPSS 9.0
    ' 02 Sep 99 - SPSS 10.0
    ' 29 mar 01 - SPSS 11.0
    ' 04 Feb 03 - SPSS 12.0 - Long variable names
    ' 27 Aug 04 - SPSS 13.0 - Extended strings
    ' 23 Aug 07 - SPSS 16.0 - Unicode

    ' Error codes returned by functions
    Public Const SPSS_OK = 0

    Public Const SPSS_FILE_OERROR = 1
    Public Const SPSS_FILE_WERROR = 2
    Public Const SPSS_FILE_RERROR = 3
    Public Const SPSS_FITAB_FULL = 4
    Public Const SPSS_INVALID_HANDLE = 5
    Public Const SPSS_INVALID_FILE = 6
    Public Const SPSS_NO_MEMORY = 7

    Public Const SPSS_OPEN_RDMODE = 8
    Public Const SPSS_OPEN_WRMODE = 9

    Public Const SPSS_INVALID_VARNAME = 10
    Public Const SPSS_DICT_EMPTY = 11
    Public Const SPSS_VAR_NOTFOUND = 12
    Public Const SPSS_DUP_VAR = 13
    Public Const SPSS_NUME_EXP = 14
    Public Const SPSS_STR_EXP = 15
    Public Const SPSS_SHORTSTR_EXP = 16
    Public Const SPSS_INVALID_VARTYPE = 17

    Public Const SPSS_INVALID_MISSFOR = 18
    Public Const SPSS_INVALID_COMPSW = 19
    Public Const SPSS_INVALID_PRFOR = 20
    Public Const SPSS_INVALID_WRFOR = 21
    Public Const SPSS_INVALID_DATE = 22
    Public Const SPSS_INVALID_TIME = 23

    Public Const SPSS_NO_VARIABLES = 24
    Public Const SPSS_MIXED_TYPES = 25
    Public Const SPSS_DUP_VALUE = 27

    Public Const SPSS_INVALID_CASEWGT = 28
    Public Const SPSS_INCOMPATIBLE_DICT = 29
    Public Const SPSS_DICT_COMMIT = 30
    Public Const SPSS_DICT_NOTCOMMIT = 31

    Public Const SPSS_NO_TYPE2 = 33
    Public Const SPSS_NO_TYPE73 = 41
    Public Const SPSS_INVALID_DATEINFO = 45
    Public Const SPSS_NO_TYPE999 = 46
    Public Const SPSS_EXC_STRVALUE = 47
    Public Const SPSS_CANNOT_FREE = 48
    Public Const SPSS_BUFFER_SHORT = 49
    Public Const SPSS_INVALID_CASE = 50
    Public Const SPSS_INTERNAL_VLABS = 51
    Public Const SPSS_INCOMPAT_APPEND = 52
    Public Const SPSS_INTERNAL_D_A = 53
    Public Const SPSS_FILE_BADTEMP = 54
    Public Const SPSS_DEW_NOFIRST = 55
    Public Const SPSS_INVALID_MEASURELEVEL = 56
    Public Const SPSS_INVALID_7SUBTYPE = 57
    Public Const SPSS_INVALID_VARHANDLE = 58
    Public Const SPSS_INVALID_ENCODING = 59
    Public Const SPSS_FILES_OPEN = 60

    Public Const SPSS_INVALID_MRSETDEF = 70
    Public Const SPSS_INVALID_MRSETNAME = 71
    Public Const SPSS_DUP_MRSETNAME = 72
    Public Const SPSS_BAD_EXTENSION = 73
    Public Const SPSS_INVALID_EXTENDEDSTRING = 74
    Public Const SPSS_INVALID_ATTRNAME = 75
    Public Const SPSS_INVALID_ATTRDEF = 76
    Public Const SPSS_INVALID_MRSETINDEX = 77
    Public Const SPSS_INVALID_VARSETDEF = 78
    Public Const SPSS_INVALID_ROLE = 79

    ' Warning codes returned by functions
    Public Const SPSS_EXC_LEN64 = -1
    Public Const SPSS_EXC_LEN120 = -2
    Public Const SPSS_EXC_VARLABEL = -2
    Public Const SPSS_EXC_LEN60 = -4
    Public Const SPSS_EXC_VALLABEL = -4
    Public Const SPSS_FILE_END = -5
    Public Const SPSS_NO_VARSETS = -6
    Public Const SPSS_EMPTY_VARSETS = -7
    Public Const SPSS_NO_LABELS = -8
    Public Const SPSS_NO_LABEL = -9
    Public Const SPSS_NO_CASEWGT = -10
    Public Const SPSS_NO_DATEINFO = -11
    Public Const SPSS_NO_MULTRESP = -12
    Public Const SPSS_EMPTY_MULTRESP = -13
    Public Const SPSS_NO_DEW = -14
    Public Const SPSS_EMPTY_DEW = -15


    ' Missing value format codes
    Public Const SPSS_NO_MISSVAL As Integer = 0
    Public Const SPSS_ONE_MISSVAL As Integer = 1
    Public Const SPSS_TWO_MISSVAL As Integer = 2
    Public Const SPSS_THREE_MISSVAL As Integer = 3
    Public Const SPSS_MISS_RANGE As Integer = -2
    Public Const SPSS_MISS_RANGEANDVAL As Integer = -3



    ' SPSS Format Type Codes
    Public Const SPSS_FMT_A As Integer = 1              ' Alphanumeric
    Public Const SPSS_FMT_AHEX As Integer = 2           ' Alphanumeric hexadecimal
    Public Const SPSS_FMT_COMMA As Integer = 3          ' F Format with commas
    Public Const SPSS_FMT_DOLLAR As Integer = 4         ' Commas and floating dollar sign
    Public Const SPSS_FMT_F As Integer = 5              ' Default Numeric Format
    Public Const SPSS_FMT_IB As Integer = 6             ' Integer binary
    Public Const SPSS_FMT_PIBHEX As Integer = 7         ' Positive integer binary - hex
    Public Const SPSS_FMT_P As Integer = 8              ' Packed decimal
    Public Const SPSS_FMT_PIB As Integer = 9            ' Positive integer binary unsigned
    Public Const SPSS_FMT_PK As Integer = 10            ' Positive integer binary unsigned
    Public Const SPSS_FMT_RB As Integer = 11            ' Floating point binary
    Public Const SPSS_FMT_RBHEX As Integer = 12         ' Floating point binary hex
    Public Const SPSS_FMT_Z As Integer = 15             ' Zoned decimal
    Public Const SPSS_FMT_N As Integer = 16             ' N Format- unsigned with leading 0s
    Public Const SPSS_FMT_E As Integer = 17             ' E Format- with explicit power of 10
    Public Const SPSS_FMT_DATE As Integer = 20          ' Date format dd-mmm-yyyy
    Public Const SPSS_FMT_TIME As Integer = 21          ' Time format hh:mm:ss.s
    Public Const SPSS_FMT_DATE_TIME As Integer = 22     ' Date and Time
    Public Const SPSS_FMT_ADATE As Integer = 23         ' Date format dd-mmm-yyyy
    Public Const SPSS_FMT_JDATE As Integer = 24         ' Julian date - yyyyddd
    Public Const SPSS_FMT_DTIME As Integer = 25         ' Date-time dd hh:mm:ss.s
    Public Const SPSS_FMT_WKDAY As Integer = 26         ' Day of the week
    Public Const SPSS_FMT_MONTH As Integer = 27         ' Month
    Public Const SPSS_FMT_MOYR As Integer = 28          ' mmm yyyy
    Public Const SPSS_FMT_QYR As Integer = 29           ' q Q yyyy
    Public Const SPSS_FMT_WKYR As Integer = 30          ' ww WK yyyy
    Public Const SPSS_FMT_PCT As Integer = 31           ' Percent - F followed by %
    Public Const SPSS_FMT_DOT As Integer = 32           ' Like COMMA, switching dot for comma
    Public Const SPSS_FMT_CCA As Integer = 33           ' User Programmable currency format
    Public Const SPSS_FMT_CCB As Integer = 34           ' User Programmable currency format
    Public Const SPSS_FMT_CCC As Integer = 35           ' User Programmable currency format
    Public Const SPSS_FMT_CCD As Integer = 36           ' User Programmable currency format
    Public Const SPSS_FMT_CCE As Integer = 37           ' User Programmable currency format
    Public Const SPSS_FMT_EDATE As Integer = 38         ' Date in dd/mm/yyyy style
    Public Const SPSS_FMT_SDATE As Integer = 39         ' Date in yyyy/mm/dd style


    ' Definitions of "type 7" records
    Public Const SPSS_T7_DOCUMENTS As Integer = 0       ' Documents (actually type 6
    Public Const SPSS_T7_VAXDE_DICT As Integer = 1      ' VAX Data Entry - dictionary version
    Public Const SPSS_T7_VAXDE_DATA As Integer = 2      ' VAX Data Entry - data
    Public Const SPSS_T7_SOURCE As Integer = 3          ' Source system characteristics
    Public Const SPSS_T7_HARDCONST As Integer = 4       ' Source system floating pt constants
    Public Const SPSS_T7_VARSETS As Integer = 5         ' Variable sets
    Public Const SPSS_T7_TRENDS As Integer = 6          ' Trends date information
    Public Const SPSS_T7_MULTRESP As Integer = 7        ' Multiple response groups
    Public Const SPSS_T7_DEW_DATA As Integer = 8        ' Windows Data Entry data
    Public Const SPSS_T7_TEXTSMART As Integer = 10      ' TextSmart data
    Public Const SPSS_T7_MSMTLEVEL As Integer = 11      ' Msmt level, col width, & alignment
    Public Const SPSS_T7_DEW_GUID As Integer = 12       ' Windows Data Entry GUID
    Public Const SPSS_T7_XVARNAMES As Integer = 13      ' Extended variable names
    Public Const SPSS_T7_XSTRINGS As Integer = 14       'Extended strings
    Public Const SPSS_T7_CLEMENTINE As Integer = 15     'Clementine Metadata
    Public Const SPSS_T7_NCASES As Integer = 16         '64 bit N of cases
    Public Const SPSS_T7_FILE_ATTR As Integer = 17      'File level attributes
    Public Const SPSS_T7_VAR_ATTR As Integer = 18       'Variable attributes
    Public Const SPSS_T7_EXTMRSETS As Integer = 19      ' Extended multiple response groups
    Public Const SPSS_T7_ENCODING As Integer = 20       ' Encoding, aka code page
    Public Const SPSS_T7_LONGSTRLABS As Integer = 21    ' Value labels for long strings
    Public Const SPSS_T7_LONGSTRMVAL As Integer = 22    ' Missing values for long strings

    ' Encoding modes
    Public Const SPSS_ENCODING_CODEPAGE As Integer = 0  ' Text encoded in current code page
    Public Const SPSS_ENCODING_UTF8 As Integer = 1      ' Text encoded as UTF-8


    ' Diagnostics regarding SPSS variable names
    Public Const SPSS_NAME_OK = 0              ' Valid standard name
    Public Const SPSS_NAME_SCRATCH = 1         ' Valid scratch var name
    Public Const SPSS_NAME_SYSTEM = 2          ' Valid system var name
    Public Const SPSS_NAME_BADLTH = 3          ' Empty or longer than SPSS_MAX_VARNAME
    Public Const SPSS_NAME_BADCHAR = 4         ' Invalid character or imbedded blank
    Public Const SPSS_NAME_RESERVED = 5        ' Name is a reserved word
    Public Const SPSS_NAME_BADFIRST = 6        ' Invalid initial character (otherwise OK)

    ' Maximum lengths of SPSS data file objects
    Public Const SPSS_MAX_VARNAME = 64         ' Variable name
    Public Const SPSS_MAX_SHORTVARNAME = 8     ' Short (compatibility) variable name
    Public Const SPSS_MAX_SHORTSTRING = 8      ' Short string variable
    Public Const SPSS_MAX_IDSTRING = 64        ' File label string
    Public Const SPSS_MAX_LONGSTRING = 32767   ' Long string variable
    Public Const SPSS_MAX_VALLABEL = 120       ' Value label
    Public Const SPSS_MAX_VARLABEL = 256       ' Variable label
    Public Const SPSS_MAX_7SUBTYPE = 40        ' Maximum record 7 subtype
    Public Const SPSS_MAX_ENCODING = 64        ' Maximum encoding text


    ' Functions exported by spssio32.dll in alphabetical order
    Public Declare Function spssAddFileAttribute Lib "spssio32.dll" Alias "spssAddFileAttribute@16" _
                            (ByVal handle As Integer, ByVal attribName As String, ByVal attribSub As Integer, ByVal attribText As String) As Integer
    ' Public Declare Function   spssAddMultRespDefC Lib "spssio32.dll" Alias "spssAddMultRespDefC@28" _
    '                           (ByVal handle As Integer, ByVal mrSetName As String, ByVal mrSetLabel As String, ByVal isDichotomy As Integer, ByVal countedValue As Integer, ByVal *varNames As String, ByVal numVars As Integer) As Integer
    ' Public Declare Function   spssAddMultRespDefExt "spssio32.dll" Alias "spssAddMultRespDefExt@8" _
    '                           (ByVal handle As Integer, ByVal pSet as String) As Integer
    ' Public Declare Function   spssAddMultRespDefN Lib "spssio32.dll" Alias "spssAddMultRespDefN@28" _
    '                           (ByVal handle As Integer, ByVal mrSetName As String, ByVal mrSetLabel As String, ByVal isDichotomy As Integer, ByVal countedValue As String, ByVal *varNames As String, ByVal numVars As Integer) As Integer
    Public Declare Function spssAddVarAttribute Lib "spssio32.dll" Alias "spssAddVarAttribute@20" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal attribName As String, ByVal attribSub As Integer, ByVal attribText As String) As Integer
    Public Declare Function spssCloseAppend Lib "spssio32.dll" Alias "spssCloseAppend@4" _
                                (ByVal handle As Integer) As Integer
    Public Declare Function spssCloseRead Lib "spssio32.dll" Alias "spssCloseRead@4" _
                                (ByVal handle As Integer) As Integer
    Public Declare Function spssCloseWrite Lib "spssio32.dll" Alias "spssCloseWrite@4" _
                                (ByVal handle As Integer) As Integer
    Public Declare Function spssCommitCaseRecord Lib "spssio32.dll" Alias "spssCommitCaseRecord@4" _
                                (ByVal handle As Integer) As Integer
    Public Declare Function spssCommitHeader Lib "spssio32.dll" Alias "spssCommitHeader@4" _
                                (ByVal handle As Integer) As Integer
    Public Declare Function spssConvertDate Lib "spssio32.dll" Alias "spssConvertDate@16" _
                                (ByVal day As Integer, ByVal month As Integer, ByVal year As Integer, ByRef spssDate As Double) As Integer
    Public Declare Function spssConvertSPSSDate Lib "spssio32.dll" Alias "spssConvertSPSSDate@20" _
                                (ByRef day As Integer, ByRef month As Integer, ByRef year As Integer, ByVal spssDate As Double) As Integer
    Public Declare Function spssConvertSPSSTime Lib "spssio32.dll" Alias "spssConvertSPSSTime@24" _
                                (ByRef day As Integer, ByRef hourh As Integer, ByRef minute As Integer, ByRef second As Double, ByVal spssDate As Double) As Integer
    Public Declare Function spssConvertTime Lib "spssio32.dll" Alias "spssConvertTime@24" _
                                (ByVal day As Integer, ByVal hour As Integer, ByVal minute As Integer, ByVal second As Double, ByRef spssTime As Double) As Integer
    Public Declare Function spssCopyDocuments Lib "spssio32.dll" Alias "spssCopyDocuments@8" _
                                (ByVal fromHandle As Integer, ByVal toHandle As Integer) As Integer
    ' Public Declare Function   spssFreeAttributes Lib "spssio32.dll" Alias "spssFreeAttributes@12" _
    '                           (ByRef *attribNames As String, ByRef *attribText As String, ByVal nAttributes) As Integer
    Public Declare Function spssFreeDateVariables Lib "spssio32.dll" Alias "spssFreeDateVariables@4" _
                                (ByRef pDateInfo As Integer) As Integer
    Public Declare Function spssFreeMultRespDefs Lib "spssio32.dll" Alias "spssFreeMultRespDefs@4" _
                                (ByVal pMrespDefs As Integer) As Integer
    ' Public Declare Function   spssFreeMultRespDefStruct Lib "spssio32.dll" Alias "spssFreeMultRespDefStruct@4" _
    '                           (ByVal pSet As String) As Integer
    ' Public Declare Function   spssFreeVarCValueLabels Lib "spssio32.dll" Alias "spssFreeVarCValueLabels@12" _
    '                           (ByRef *values As String, ByRef *labels As String, ByVal numLabels As Integer) As Integer
    Public Declare Function spssFreeVariableSets Lib "spssio32.dll" Alias "spssFreeVariableSets@4" _
                                (ByVal pVarSets As Integer) As Integer
    ' Public Declare Function   spssFreeVarNames Lib "spssio32.dll" Alias "spssFreeVarNames@12" _
    '                           (ByVal *varNames As String, ByRef varTypes As Integer, ByVal numVars As Integer) As Integer
    ' Public Declare Function   spssFreeVarNValueLabels Lib "spssio32.dll" Alias "spssFreeVarNValueLabels@12" _
    '                           (ByRef values As Double, ByVal *labels As String, ByVal numLabels As Integer) As Integer
    Public Declare Function spssGetCaseSize Lib "spssio32.dll" Alias "spssGetCaseSize@8" _
                                (ByVal handle As Integer, ByRef caseSize As Integer) As Integer
    Public Declare Function spssGetCaseWeightVar Lib "spssio32.dll" Alias "spssGetCaseWeightVar@8" _
                                (ByVal handle As Integer, ByVal varName As String) As Integer
    Public Declare Function spssGetCompression Lib "spssio32.dll" Alias "spssGetCompression@8" _
                                (ByVal handle As Integer, ByRef compSwitch As Integer) As Integer
    Public Declare Function spssGetDateVariables Lib "spssio32.dll" Alias "spssGetDateVariables@12" _
                                (ByVal handle As Integer, ByRef numofElements As Integer, ByRef pDateInfo As Integer) As Integer
    Public Declare Function spssGetDEWFirst Lib "spssio32.dll" Alias "spssGetDEWFirst@16" _
                                (ByVal handle As Integer, ByVal Data As String, ByVal maxData As Integer, ByRef Data As Integer) As Integer
    Public Declare Function spssGetDEWGUID Lib "spssio32.dll" Alias "spssGetDEWGUID@8" _
                                (ByVal handle As Integer, ByVal asciiGUID As String) As Integer
    Public Declare Function spssGetDEWInfo Lib "spssio32.dll" Alias "spssGetDEWInfo@12" _
                                (ByVal handle As Integer, ByRef Length As Integer, ByRef HashTotal As Integer) As Integer
    Public Declare Function spssGetDEWNext Lib "spssio32.dll" Alias "spssGetDEWNext@12" _
                                (ByVal handle As Integer, ByVal Data As String, ByVal maxData As Integer, ByRef nData As Integer) As Integer
    Public Declare Function spssGetEstimatedNofCases Lib "spssio32.dll" Alias "spssGetEstimatedNofCases@8" _
                                (ByVal handle As Integer, ByRef caseCount As Integer) As Integer
    ' Public Declare Function   spssGetFileAttributes Lib "spssio32.dll" Alias "spssGetFileAttributes@16" _
    '                           (ByVal handle As Integer, ByVal **attribNames As String, ByVal **attribText As String, ByRef nAttributes As Integer) As Integer
    Public Declare Function spssGetFileCodePage Lib "spssio32.dll" Alias "spssGetFileCodePage@8" _
                                (ByVal handle As Integer, ByRef nCodePage As Integer) As Integer
    Public Declare Function spssGetFileEncoding Lib "spssio32.dll" Alias "spssGetFileEncoding@8" _
                                (ByVal handle As Integer, ByVal szEncoding As String) As Integer
    Public Declare Function spssGetIdString Lib "spssio32.dll" Alias "spssGetIdString@8" _
                                (ByVal handle As Integer, ByVal id As String) As Integer
    Public Declare Function spssGetInterfaceEncoding Lib "spssio32.dll" Alias "spssGetInterfaceEncoding@0" _
                                () As Integer
    Public Declare Function spssGetMultRespCount Lib "spssio32.dll" Alias "spssGetMultRespCount@8" _
                                (ByVal handle As Integer, ByRef nSets As Integer) As Integer
    ' Public Declare Function   spssGetMultRespDefByIndex Lib "spssio32.dll" Alias "spssGetMultRespDefByIndex@12" _
    '                           (ByVal handle As Integer, ByVal iSet As Integer, ByVal *ppSet As String) As Integer
    Public Declare Function spssGetMultRespDefs Lib "spssio32.dll" Alias "spssGetMultRespDefs@8" _
                                (ByVal handle As Integer, ByRef pMrespDefs As Integer) As Integer
    Public Declare Function spssGetMultRespDefsEx Lib "spssio32.dll" Alias "spssGetMultRespDefsEx@8" _
                                (ByVal handle As Integer, ByRef pMrespDefs As Integer) As Integer
    Public Declare Function spssGetNumberofCases Lib "spssio32.dll" Alias "spssGetNumberofCases@8" _
                                (ByVal handle As Integer, ByRef caseCount As Integer) As Integer
    Public Declare Function spssGetNumberofVariables Lib "spssio32.dll" Alias "spssGetNumberofVariables@8" _
                                (ByVal handle As Integer, ByRef numVars As Integer) As Integer
    Public Declare Function spssGetReleaseInfo Lib "spssio32.dll" Alias "spssGetReleaseInfo@8" _
                                (ByVal handle As Integer, ByRef relInfo As Integer) As Integer
    Public Declare Function spssGetSystemString Lib "spssio32.dll" Alias "spssGetSystemString@8" _
                                (ByVal handle As Integer, ByVal sysName As String) As Integer
    Public Declare Function spssGetTextInfo Lib "spssio32.dll" Alias "spssGetTextInfo@8" _
                                (ByVal handle As Integer, ByVal textInfo As String) As Integer
    Public Declare Function spssGetTimeStamp Lib "spssio32.dll" Alias "spssGetTimeStamp@12" _
                                (ByVal handle As Integer, ByVal fileDate As String, ByVal fileTime As String) As Integer
    Public Declare Function spssGetValueChar Lib "spssio32.dll" Alias "spssGetValueChar@20" _
                                (ByVal handle As Integer, ByVal varHandle As Double, ByVal value As String, ByVal valueSize As Integer) As Integer
    Public Declare Function spssGetValueNumeric Lib "spssio32.dll" Alias "spssGetValueNumeric@16" _
                                (ByVal handle As Integer, ByVal varHandle As Double, ByRef value As Double) As Integer
    Public Declare Function spssGetVarAlignment Lib "spssio32.dll" Alias "spssGetVarAlignment@12" _
                                (ByVal handle As Integer, ByVal varName As String, ByRef alignment As Integer) As Integer
    ' Public Declare Function   spssGetVarAttributes Lib "spssio32.dll" Alias "spssGetVarAttributes@20" _
    '                           (ByVal handle As Integer, ByVal varName As String, ByVal **attribNames As String, ByVal **attribText As String, ByRef nAttributes As Integer) As Integer
    Public Declare Function spssGetVarCMissingValues Lib "spssio32.dll" Alias "spssGetVarCMissingValues@24" _
                                (ByVal handle As Integer, ByVal varName As String, ByRef missingFormat As Integer, ByVal missingVal1 As String, ByVal missingVal2 As String, ByVal missingVal3 As String) As Integer
    Public Declare Function spssGetVarColumnWidth Lib "spssio32.dll" Alias "spssGetVarColumnWidth@12" _
                                (ByVal handle As Integer, ByVal varName As String, ByRef columnWidth As Integer) As Integer
    Public Declare Function spssGetVarCompatName Lib "spssio32.dll" Alias "spssGetVarCompatName@12" _
                                (ByVal handle As Integer, ByVal longName As String, ByVal shortName As String) As Integer
    Public Declare Function spssGetVarCValueLabel Lib "spssio32.dll" Alias "spssGetVarCValueLabel@16" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal value As String, ByVal label As String) As Integer
    Public Declare Function spssGetVarCValueLabelLong Lib "spssio32.dll" Alias "spssGetVarCValueLabelLong@24" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal value As String, ByVal labelBuff As String, ByVal lenBuff As Integer, ByRef lenLabel As Integer) As Integer
    ' Public Declare Function   spssGetVarCValueLabels Lib "spssio32.dll" Alias "spssGetVarCValueLabels@20" _
    '                           (ByVal handle As Integer, ByVal varName As String, ByVal **values As String, ByVal **labels As String, ByRef numofLabels As Integer) As Integer
    Public Declare Function spssGetVarHandle Lib "spssio32.dll" Alias "spssGetVarHandle@12" _
                                (ByVal handle As Integer, ByVal varName As String, ByRef varHandle As Double) As Integer
    Public Declare Function spssGetVariableSets Lib "spssio32.dll" Alias "spssGetVariableSets@8" _
                                (ByVal handle As Integer, ByRef pVarSets As Integer) As Integer
    Public Declare Function spssGetVarInfo Lib "spssio32.dll" Alias "spssGetVarInfo@16" _
                                (ByVal handle As Integer, ByVal iVar As Integer, ByVal varName As String, ByRef varType As Integer) As Integer
    Public Declare Function spssGetVarLabel Lib "spssio32.dll" Alias "spssGetVarLabel@12" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal varLabel As String) As Integer
    Public Declare Function spssGetVarLabelLong Lib "spssio32.dll" Alias "spssGetVarLabelLong@20" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal labelBuff As String, ByVal lenBuff As Integer, ByRef lenLabel As Integer) As Integer
    Public Declare Function spssGetVarMeasureLevel Lib "spssio32.dll" Alias "spssGetVarMeasureLevel@12" _
                                (ByVal handle As Integer, ByVal varName As String, ByRef measureLevel As Integer) As Integer
    Public Declare Function spssGetVarRole Lib "spssio32.dll" Alias "spssGetVarRole@12" _
                            (ByVal handle As Integer, ByVal varName As String, ByRef varRole As Integer) As Integer
    ' Public Declare Function   spssGetVarNames Lib "spssio32.dll" Alias "spssGetVarNames@16" _
    '                           (ByVal handle As Integer, ByRef numVars As Integer, ByRef **varNames As String, ByRef *varTypes As Integer) As Integer
    Public Declare Function spssGetVarNMissingValues Lib "spssio32.dll" Alias "spssGetVarNMissingValues@24" _
                                (ByVal handle As Integer, ByVal varName As String, ByRef missingFormat As Integer, ByRef missingVal1 As Double, ByRef missingVal2 As Double, ByRef missingVal3 As Double) As Integer
    Public Declare Function spssGetVarNValueLabel Lib "spssio32.dll" Alias "spssGetVarNValueLabel@20" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal value As Double, ByVal label As String) As Integer
    Public Declare Function spssGetVarNValueLabelLong Lib "spssio32.dll" Alias "spssGetVarNValueLabelLong@28" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal value As Double, ByVal labelBuff As String, ByVal lenBuff As Integer, ByRef lenLabel As Integer) As Integer
    ' Public Declare Function   spssGetVarNValueLabels Lib "spssio32.dll" Alias "spssGetVarNValueLabels@20" _
    '                           (ByVal handle As Integer, ByVal varName As String, ByRef *values As Double, ByVal **labels As String, ByRef numofLabels As Integer) As Integer
    Public Declare Function spssGetVarPrintFormat Lib "spssio32.dll" Alias "spssGetVarPrintFormat@20" _
                                (ByVal handle As Integer, ByVal varName As String, ByRef printType As Integer, ByRef printDec As Integer, ByRef printWidth As Integer) As Integer
    Public Declare Function spssGetVarWriteFormat Lib "spssio32.dll" Alias "spssGetVarWriteFormat@20" _
                                (ByVal handle As Integer, ByVal varName As String, ByRef writeType As Integer, ByRef writeDec As Integer, ByRef writeWidth As Integer) As Integer
    Public Declare Sub spssHostSysmisVal Lib "spssio32.dll" Alias "spssHostSysmisVal@4" _
                                (ByRef missVal As Double)
    Public Declare Function spssIsCompatibleEncoding Lib "spssio32.dll" Alias "spssIsCompatibleEncoding@8" _
                                (ByVal handle As Integer, ByRef bCompatible As Integer) As Integer
    Public Declare Sub spssLowHighVal Lib "spssio32.dll" Alias "spssLowHighVal@8" _
                                (ByRef lowest As Double, ByRef highest As Double)
    Public Declare Function spssOpenAppend Lib "spssio32.dll" Alias "spssOpenAppend@8" _
                                (ByVal fileName As String, ByRef handle As Integer) As Integer
    Public Declare Function spssOpenRead Lib "spssio32.dll" Alias "spssOpenRead@8" _
                                (ByVal fileName As String, ByRef handle As Integer) As Integer
    Public Declare Function spssOpenWrite Lib "spssio32.dll" Alias "spssOpenWrite@8" _
                                (ByVal fileName As String, ByRef handle As Integer) As Integer
    Public Declare Function spssOpenWriteCopy Lib "spssio32.dll" Alias "spssOpenWriteCopy@12" _
                                (ByVal fileName As String, ByVal dictFileName As String, ByRef handle As Integer) As Integer
    Public Declare Function spssQueryType7 Lib "spssio32.dll" Alias "spssQueryType7@12" _
                                (ByVal fromHandle As Integer, ByVal subType As Integer, ByRef bFound As Integer) As Integer
    Public Declare Function spssReadCaseRecord Lib "spssio32.dll" Alias "spssReadCaseRecord@4" _
                                (ByVal handle As Integer) As Integer
    Public Declare Function spssSeekNextCase Lib "spssio32.dll" Alias "spssSeekNextCase@8" _
                                (ByVal handle As Integer, ByVal caseNumber As Integer) As Integer
    Public Declare Function spssSetCaseWeightVar Lib "spssio32.dll" Alias "spssSetCaseWeightVar@8" _
                                (ByVal handle As Integer, ByVal varName As String) As Integer
    Public Declare Function spssSetCompression Lib "spssio32.dll" Alias "spssSetCompression@8" _
                                (ByVal handle As Integer, ByVal compSwitch As Integer) As Integer
    Public Declare Function spssSetDateVariables Lib "spssio32.dll" Alias "spssSetDateVariables@12" _
                                (ByVal handle As Integer, ByVal numofElements As Integer, ByRef dateInfo As Integer) As Integer
    Public Declare Function spssSetDEWFirst Lib "spssio32.dll" Alias "spssSetDEWFirst@12" _
                                (ByVal handle As Integer, ByVal Data As String, ByVal nBytes As Integer) As Integer
    Public Declare Function spssSetDEWGUID Lib "spssio32.dll" Alias "spssSetDEWGUID@8" _
                                (ByVal handle As Integer, ByVal asciiGUID As String) As Integer
    Public Declare Function spssSetDEWNext Lib "spssio32.dll" Alias "spssSetDEWNext12" _
                                (ByVal handle As Integer, ByVal Data As String, ByVal nBytes As Integer) As Integer
    ' Public Declare Function   spssSetFileAttributes Lib "spssio32.dll" Alias "spssSetFileAttributes@16" _
    '                           (ByVal handle As Integer, ByVal *attribNames As String, ByVal *attribText As String, ByVal nAttributes As Integer) As Integer
    Public Declare Function spssSetIdString Lib "spssio32.dll" Alias "spssSetIdString@8" _
                                (ByVal handle As Integer, ByVal id As String) As Integer
    Public Declare Function spssSetInterfaceEncoding Lib "spssio32.dll" Alias "spssSetInterfaceEncoding@4" _
                                (ByVal iEncoding As Integer) As Integer
    ' Public Declare Function   spssSetLocale Lib "spssio32.dll" Alias "spssSetLocale@8" _
    '                           (ByVal iCategory As Integer, ByVal szLocale As String) As String
    Public Declare Function spssSetMultRespDefs Lib "spssio32.dll" Alias "spssSetMultRespDefs@8" _
                                (ByVal handle As Integer, ByVal mrespDefs As String) As Integer
    Public Declare Function spssSetTempDir Lib "spssio32.dll" Alias "spssSetTempDir@4" _
                                (ByVal dirName As String) As Integer
    Public Declare Function spssSetTextInfo Lib "spssio32.dll" Alias "spssSetTextInfo@8" _
                                (ByVal handle As Integer, ByVal textInfo As String) As Integer
    Public Declare Function spssSetValueChar Lib "spssio32.dll" Alias "spssSetValueChar@16" _
                                (ByVal handle As Integer, ByVal varHandle As Double, ByVal value As String) As Integer
    Public Declare Function spssSetValueNumeric Lib "spssio32.dll" Alias "spssSetValueNumeric@20" _
                                (ByVal handle As Integer, ByVal varHandle As Double, ByVal value As Double) As Integer
    Public Declare Function spssSetVarAlignment Lib "spssio32.dll" Alias "spssSetVarAlignment@12" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal alignment As Integer) As Integer
    ' Public Declare Function   spssSetVarAttributes Lib "spssio32.dll" Alias "spssSetVarAttributes@20" _
    '                           (ByVal handle As Integer, ByVal varName As String, ByVal *attribNames As String, ByVal *attribText As String, ByVal nAttributes As Integer) As Integer
    Public Declare Function spssSetVarCMissingValues Lib "spssio32.dll" Alias "spssSetVarCMissingValues@24" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal missingFormat As Integer, ByVal missingVal1 As String, ByVal missingVal2 As String, ByVal missingVal3 As String) As Integer
    Public Declare Function spssSetVarColumnWidth Lib "spssio32.dll" Alias "spssSetVarColumnWidth@12" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal columnWidth As Integer) As Integer
    Public Declare Function spssSetVarCValueLabel Lib "spssio32.dll" Alias "spssSetVarCValueLabel@16" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal value As String, ByVal label As String) As Integer
    ' Public Declare Function   spssSetVarCValueLabels Lib "spssio32.dll" Alias "spssSetVarCValueLabels@20" _
    '                           (ByVal handle As Integer, ByVal *varNames As String, ByVal numofVars As Integer, ByVal *values As String, ByVal *labels As String, ByVal numofLabels As Integer) As Integer
    Public Declare Function spssSetVariableSets Lib "spssio32.dll" Alias "spssSetVariableSets@8" _
                                (ByVal handle As Integer, ByVal varSets As String) As Integer
    Public Declare Function spssSetVarLabel Lib "spssio32.dll" Alias "spssSetVarLabel@12" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal varLabel As String) As Integer
    Public Declare Function spssSetVarMeasureLevel Lib "spssio32.dll" Alias "spssSetVarMeasureLevel@12" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal measureLevel As Integer) As Integer
    Public Declare Function spssSetVarRole Lib "spssio32.dll" Alias "spssSetVarRole@12" _
                            (ByVal handle As Integer, ByVal varName As String, ByVal varRole As Integer) As Integer
    Public Declare Function spssSetVarName Lib "spssio32.dll" Alias "spssSetVarName@12" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal varType As Integer) As Integer
    Public Declare Function spssSetVarNMissingValues Lib "spssio32.dll" Alias "spssSetVarNMissingValues@36" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal missingFormat As Integer, ByVal missingVal1 As Double, ByVal missingVal2 As Double, ByVal missingVal3 As Double) As Integer
    Public Declare Function spssSetVarNValueLabel Lib "spssio32.dll" Alias "spssSetVarNValueLabel@20" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal value As Double, ByVal label As String) As Integer
    ' Public Declare Function   spssSetVarNValueLabels Lib "spssio32.dll" Alias "spssSetVarNValueLabels@24" _
    '                           (ByVal handle As Integer, ByVal *varNames As String, ByVal numofVars As Integer, ByRef values As Double, ByVal *labels As String, ByVal numofLabels As Integer) As Integer
    Public Declare Function spssSetVarPrintFormat Lib "spssio32.dll" Alias "spssSetVarPrintFormat@20" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal printType As Integer, ByVal printDec As Integer, ByVal printWidth As Integer) As Integer
    Public Declare Function spssSetVarWriteFormat Lib "spssio32.dll" Alias "spssSetVarWriteFormat@20" _
                                (ByVal handle As Integer, ByVal varName As String, ByVal writeType As Integer, ByVal writeDec As Integer, ByVal writeWidth As Integer) As Integer
    Public Declare Function spssSysmisVal Lib "spssio32.dll" Alias "spssSysmisVal@0" _
                                () As Double
    Public Declare Function spssValidateVarname Lib "spssio32.dll" Alias "spssValidateVarname@4" _
                                (ByVal varName As String) As Integer
    Public Declare Function spssWholeCaseIn Lib "spssio32.dll" Alias "spssWholeCaseIn@8" _
                                (ByVal handle As Integer, ByRef caseRec As Double) As Integer
    Public Declare Function spssWholeCaseOut Lib "spssio32.dll" Alias "spssWholeCaseOut@8" _
                                (ByVal handle As Integer, ByRef caseRec As Double) As Integer
End Module

