/***********************************************************************
 * Licensed Materials - Property of IBM 
 *
 * IBM SPSS Products: Statistics Common
 *
 * (C) Copyright IBM Corp. 1989, 2012
 *
 * US Government Users Restricted Rights - Use, duplication or disclosure
 * restricted by GSA ADP Schedule Contract with IBM Corp. 
 ************************************************************************/

/*
**++
**  NAME
**    spssdiocodes.h - Codes for the SPSS Data File I/O API.
**
**  DESCRIPTION
**    This is part of spssdio.h.  It declares only the codes, not
**    the entries, in that interface.
**--
**  COPYRIGHT
**    (c) Copyright 2005 by SPSS Inc.
**
**  Modifications:
**
**  06 Jan 05 - Fry     - extracted from spssdio.h
**  20 Jan 11 - Weijun - add new ML SPSS_MLVL_FLA,  SPSS_MLVL_TPL
**  31 Mar 11 - Weijun - add new role SPSS_ROLE_FREQWEIGHT
**  05 Jan 12 - Weijun - add new role SPSS_ROLE_RECORD_ID
*/

#ifndef SPSSDIOCODES_H
#define SPSSDIOCODES_H

/****************************************** VARIABLE TYPES
*/
#define    SPSS_STRING(length)        (length)
#define    SPSS_NUMERIC               0

/****************************************** MISSING VALUE TYPE CODES
*/

#define    SPSS_NO_MISSVAL            0
#define    SPSS_ONE_MISSVAL           1
#define    SPSS_TWO_MISSVAL           2
#define    SPSS_THREE_MISSVAL         3
#define    SPSS_MISS_RANGE           -2
#define    SPSS_MISS_RANGEANDVAL     -3

/****************************************** ERROR CODES
*/
#define    SPSS_OK                    0

#define    SPSS_FILE_OERROR           1
#define    SPSS_FILE_WERROR           2
#define    SPSS_FILE_RERROR           3
#define    SPSS_FITAB_FULL            4
#define    SPSS_INVALID_HANDLE        5
#define    SPSS_INVALID_FILE          6
#define    SPSS_NO_MEMORY             7

#define    SPSS_OPEN_RDMODE           8
#define    SPSS_OPEN_WRMODE           9

#define    SPSS_INVALID_VARNAME       10
#define    SPSS_DICT_EMPTY            11
#define    SPSS_VAR_NOTFOUND          12
#define    SPSS_DUP_VAR               13
#define    SPSS_NUME_EXP              14
#define    SPSS_STR_EXP               15
#define    SPSS_SHORTSTR_EXP          16
#define    SPSS_INVALID_VARTYPE       17

#define    SPSS_INVALID_MISSFOR       18
#define    SPSS_INVALID_COMPSW        19
#define    SPSS_INVALID_PRFOR         20
#define    SPSS_INVALID_WRFOR         21
#define    SPSS_INVALID_DATE          22
#define    SPSS_INVALID_TIME          23

#define    SPSS_NO_VARIABLES          24
#define    SPSS_MIXED_TYPES           25
#define    SPSS_DUP_VALUE             27

#define    SPSS_INVALID_CASEWGT       28
#define    SPSS_INCOMPATIBLE_DICT     29
#define    SPSS_DICT_COMMIT           30
#define    SPSS_DICT_NOTCOMMIT        31

#define    SPSS_NO_TYPE2              33
#define    SPSS_NO_TYPE73             41
#define    SPSS_INVALID_DATEINFO      45
#define    SPSS_NO_TYPE999            46
#define    SPSS_EXC_STRVALUE          47
#define    SPSS_CANNOT_FREE           48
#define    SPSS_BUFFER_SHORT          49
#define    SPSS_INVALID_CASE          50
#define    SPSS_INTERNAL_VLABS        51
#define    SPSS_INCOMPAT_APPEND       52
#define    SPSS_INTERNAL_D_A          53
#define    SPSS_FILE_BADTEMP          54
#define    SPSS_DEW_NOFIRST           55
#define    SPSS_INVALID_MEASURELEVEL  56
#define    SPSS_INVALID_7SUBTYPE      57
#define    SPSS_INVALID_VARHANDLE     58
#define    SPSS_INVALID_ENCODING      59
#define    SPSS_FILES_OPEN            60

#define    SPSS_INVALID_MRSETDEF      70
#define    SPSS_INVALID_MRSETNAME     71
#define    SPSS_DUP_MRSETNAME         72

#define    SPSS_BAD_EXTENSION         73
#define    SPSS_INVALID_EXTENDEDSTRING 74

#define    SPSS_INVALID_ATTRNAME      75
#define    SPSS_INVALID_ATTRDEF       76
#define    SPSS_INVALID_MRSETINDEX    77
#define    SPSS_INVALID_VARSETDEF     78
#define    SPSS_INVALID_ROLE          79

/****************************************** WARNING CODES
*/
#define    SPSS_EXC_LEN64            -1
#define    SPSS_EXC_LEN120           -2     /* Retain for compatibility */
#define    SPSS_EXC_VARLABEL         -2
#define    SPSS_EXC_LEN60            -4
#define    SPSS_EXC_VALLABEL         -4
#define    SPSS_FILE_END             -5
#define    SPSS_NO_VARSETS           -6
#define    SPSS_EMPTY_VARSETS        -7
#define    SPSS_NO_LABELS            -8
#define    SPSS_NO_LABEL             -9
#define    SPSS_NO_CASEWGT          -10
#define    SPSS_NO_DATEINFO         -11
#define    SPSS_NO_MULTRESP         -12
#define    SPSS_EMPTY_MULTRESP      -13
#define    SPSS_NO_DEW              -14
#define    SPSS_EMPTY_DEW           -15

/****************************************** FORMAT TYPE CODES
*/
#define   SPSS_FMT_A            1     /* Alphanumeric */
#define   SPSS_FMT_AHEX         2     /* Alphanumeric hexadecimal */
#define   SPSS_FMT_COMMA        3     /* F Format with commas */
#define   SPSS_FMT_DOLLAR       4     /* Commas and floating dollar sign */
#define   SPSS_FMT_F            5     /* Default Numeric Format */
#define   SPSS_FMT_IB           6     /* Integer binary */
#define   SPSS_FMT_PIBHEX       7     /* Positive integer binary - hex */
#define   SPSS_FMT_P            8     /* Packed decimal */
#define   SPSS_FMT_PIB          9     /* Positive integer binary unsigned */
#define   SPSS_FMT_PK          10     /* Positive integer binary unsigned */
#define   SPSS_FMT_RB          11     /* Floating point binary */
#define   SPSS_FMT_RBHEX       12     /* Floating point binary hex */
#define   SPSS_FMT_Z           15     /* Zoned decimal */
#define   SPSS_FMT_N           16     /* N Format- unsigned with leading 0s */
#define   SPSS_FMT_E           17     /* E Format- with explicit power of 10 */
#define   SPSS_FMT_DATE        20     /* Date format dd-mmm-yyyy */
#define   SPSS_FMT_TIME        21     /* Time format hh:mm:ss.s  */
#define   SPSS_FMT_DATE_TIME   22     /* Date and Time           */
#define   SPSS_FMT_ADATE       23     /* Date format mm/dd/yyyy */
#define   SPSS_FMT_JDATE       24     /* Julian date - yyyyddd   */
#define   SPSS_FMT_DTIME       25     /* Date-time dd hh:mm:ss.s */
#define   SPSS_FMT_WKDAY       26     /* Day of the week         */
#define   SPSS_FMT_MONTH       27     /* Month                   */
#define   SPSS_FMT_MOYR        28     /* mmm yyyy                */
#define   SPSS_FMT_QYR         29     /* q Q yyyy                */
#define   SPSS_FMT_WKYR        30     /* ww WK yyyy              */
#define   SPSS_FMT_PCT         31     /* Percent - F followed by % */
#define   SPSS_FMT_DOT         32     /* Like COMMA, switching dot for comma */
#define   SPSS_FMT_CCA         33     /* User Programmable currency format   */
#define   SPSS_FMT_CCB         34     /* User Programmable currency format   */
#define   SPSS_FMT_CCC         35     /* User Programmable currency format   */
#define   SPSS_FMT_CCD         36     /* User Programmable currency format   */
#define   SPSS_FMT_CCE         37     /* User Programmable currency format   */
#define   SPSS_FMT_EDATE       38     /* Date in dd.mm.yyyy style            */
#define   SPSS_FMT_SDATE       39     /* Date in yyyy/mm/dd style            */

/****************************************** MEASUREMENT LEVEL CODES
*/
#define   SPSS_MLVL_UNK         0     /* Unknown */
#define   SPSS_MLVL_NOM         1     /* Nominal */
#define   SPSS_MLVL_ORD         2     /* Ordinal */
#define   SPSS_MLVL_RAT         3     /* Scale (Ratio) (Continues) */
#define   SPSS_MLVL_FLA          4     /* Flag */
#define   SPSS_MLVL_TPL          5     /* Typeless */

/****************************************** ALIGNMENT CODES
*/
#define   SPSS_ALIGN_LEFT       0
#define   SPSS_ALIGN_RIGHT      1
#define   SPSS_ALIGN_CENTER     2

/****************************************** ROLE CODES
*/
#define   SPSS_ROLE_INPUT       0     /* Input Role */
#define   SPSS_ROLE_TARGET      1     /* Target Role */
#define   SPSS_ROLE_BOTH        2     /* Both Roles */
#define   SPSS_ROLE_NONE        3     /* None Role */
#define   SPSS_ROLE_PARTITION   4     /* Partition Role */
#define   SPSS_ROLE_SPLIT       5     /* Split Role */
#define   SPSS_ROLE_FREQUENCY       6     /* Frequency Role */
#define   SPSS_ROLE_RECORDID           7     /* Record ID */

/****************************************** DIAGNOSTICS REGARDING VAR NAMES
*/
#define SPSS_NAME_OK        0   /* Valid standard name */
#define SPSS_NAME_SCRATCH   1   /* Valid scratch var name */
#define SPSS_NAME_SYSTEM    2   /* Valid system var name */
#define SPSS_NAME_BADLTH    3   /* Empty or longer than SPSS_MAX_VARNAME */
#define SPSS_NAME_BADCHAR   4   /* Invalid character or imbedded blank */
#define SPSS_NAME_RESERVED  5   /* Name is a reserved word */
#define SPSS_NAME_BADFIRST  6   /* Invalid initial character */

/****************************************** MAXIMUM LENGTHS OF DATA FILE OBJECTS
*/
#define   SPSS_MAX_VARNAME     64     /* Variable name */
#define   SPSS_MAX_SHORTVARNAME 8     /* Short (compatibility) variable name */
#define   SPSS_MAX_SHORTSTRING  8     /* Short string variable */
#define   SPSS_MAX_IDSTRING    64     /* File label string */
#define   SPSS_MAX_LONGSTRING 32767   /* Long string variable */
#define   SPSS_MAX_VALLABEL   120     /* Value label */
#define   SPSS_MAX_VARLABEL   256     /* Variable label */
#define   SPSS_MAX_ENCODING    64     /* Maximum encoding text */
#define   SPSS_MAX_7SUBTYPE    40     /* Maximum record 7 subtype */

/****************************************** Type 7 record subtypes
*/
#define   SPSS_T7_DOCUMENTS     0     /* Documents (actually type 6 */
#define   SPSS_T7_VAXDE_DICT    1     /* VAX Data Entry - dictionary version */
#define   SPSS_T7_VAXDE_DATA    2     /* VAX Data Entry - data */
#define   SPSS_T7_SOURCE        3     /* Source system characteristics */
#define   SPSS_T7_HARDCONST     4     /* Source system floating pt constants */
#define   SPSS_T7_VARSETS       5     /* Variable sets */
#define   SPSS_T7_TRENDS        6     /* Trends date information */
#define   SPSS_T7_MULTRESP      7     /* Multiple response groups */
#define   SPSS_T7_DEW_DATA      8     /* Windows Data Entry data */
#define   SPSS_T7_TEXTSMART    10     /* TextSmart data */
#define   SPSS_T7_MSMTLEVEL    11     /* Msmt level, col width, & alignment */
#define   SPSS_T7_DEW_GUID     12     /* Windows Data Entry GUID */
#define   SPSS_T7_XVARNAMES    13     /* Extended variable names */
#define   SPSS_T7_XSTRINGS     14     /* Extended strings */
#define   SPSS_T7_CLEMENTINE   15     /* Clementine Metadata */
#define   SPSS_T7_NCASES       16     /* 64 bit N of cases */
#define   SPSS_T7_FILE_ATTR    17     /* File level attributes */
#define   SPSS_T7_VAR_ATTR     18     /* Variable attributes */
#define   SPSS_T7_EXTMRSETS    19     /* Extended multiple response groups */
#define   SPSS_T7_ENCODING     20     /* Encoding, aka code page */
#define   SPSS_T7_LONGSTRLABS  21     /* Value labels for long strings */
#define   SPSS_T7_LONGSTRMVAL  22     /* Missing values for long strings */
#define   SPSS_T7_SORTINDEX    23     /* Sort Index information */

/****************************************** Encoding modes
*/
#define   SPSS_ENCODING_CODEPAGE    0 /* Text encoded in current code page */
#define   SPSS_ENCODING_UTF8        1 /* Text encoded as UTF-8 */

/****************************************** API FUNCTIONS IN ALPHABETIC ORDER
*/

#endif
