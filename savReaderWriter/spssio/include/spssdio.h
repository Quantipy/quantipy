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
**    spssdio.h - Interface header for the SPSS Data File I/O API.
**
**  DESCRIPTION
**    Code which calls any SPSS Data File I/O API functions must include this
**    header
**--
**  COPYRIGHT
**    (c) Copyright 2002 by SPSS Inc.
**
**  Modifications:
**
**  04 May 2004 - Bartley - longer strings
**  29 Dec 2004 - Bartley - longer value labels
**  10 Feb 2005 - mholubow - reduced max value label limit from 255 to 120
*/

#ifndef SPSSDIO_H
#define SPSSDIO_H

#include "spssdiocodes.h"       /* codes are in a separate file */

#if defined(__cplusplus)
    extern "C" {
#endif

/* We need __stdcall for Windows but not for Unix
*/
#if (defined(_MSC_VER) && !defined(_WIN32)) || \
      (defined(__BORLANDC__) && !defined(__WIN32__))
    /* Looks like 16 bit Windows */
#   error The 16 bit Windows environment is no longer supported
#elif defined(_WIN32) || defined(__WIN32__)
    /* 32 bit Windows - don't do anything */
#else
    /* outside Windows */
#   define __stdcall
#endif

/* Identify exported symbols to GCC */
#if defined BUILDING_SPSSDIO
#  if (defined __GNUC__ && __GNUC__ >= 4)
       /* Identify exported symbols for the GCC compiler */
#      define SPSSDIO_API __attribute__ ((visibility ("default")))
#  elif (defined __SUNPRO_CC && __SUNPRO_CC >= 0x550)
       /* Identify exported symbols for the Sun compiler */
#      define SPSSDIO_API __global
#  elif (defined __HP_aCC && __HP_aCC >= 062000)
       /* Identify exported symbols for the HP compiler */
#      define SPSSDIO_API __attribute__ ((visibility ("default")))
#  else
#    define SPSSDIO_API
#  endif
#else
#  define SPSSDIO_API
#endif

/*  For describing one multiple response set */
typedef struct spssMultRespDef_T
{
    char szMrSetName[SPSS_MAX_VARNAME+1];  /* Null-terminated MR set name */
    char szMrSetLabel[SPSS_MAX_VARLABEL+1];  /* Null-terminated set label */
    int qIsDichotomy;              /* Whether a multiple dichotomy set */
    int qIsNumeric;                /* Whether the counted value is numeric */
    int qUseCategoryLabels;        /* Whether to use var category labels */
    int qUseFirstVarLabel;         /* Whether using var label as set label */
    int Reserved[14];              /* Reserved for expansion */
    long nCountedValue;            /* Counted value if numeric */
    char* pszCountedValue;         /* Null-terminated counted value if string */
    char** ppszVarNames;           /* Vector of null-terminated var names */
    int nVariables;                /* Number of constituent variables */
} spssMultRespDef;


/****************************************** API FUNCTIONS IN ALPHABETIC ORDER
*/

SPSSDIO_API int __stdcall spssAddFileAttribute(
    const int hFile,
    const char* attribName,
    const int attribSub,
    const char* attribText);

SPSSDIO_API int __stdcall spssAddMultRespDefC(
    const int hFile,
    const char* mrSetName,
    const char* mrSetLabel,
    const int isDichotomy,
    const char* countedValue,
    const char* * varNames,
    const int numVars);

SPSSDIO_API int __stdcall spssAddMultRespDefExt(
    const int hFile,
    const spssMultRespDef* pSet);

SPSSDIO_API int __stdcall spssAddMultRespDefN(
    const int hFile,
    const char* mrSetName,
    const char* mrSetLabel,
    const int isDichotomy,
    const long countedValue,
    const char* * varNames,
    const int numVars);

SPSSDIO_API int __stdcall spssAddVarAttribute(
    const int hFile,
    const char* varName,
    const char* attribName,
    const int attribSub,
    const char* attribText);

SPSSDIO_API int __stdcall spssCloseAppend(
    const int hFile);

SPSSDIO_API int __stdcall spssCloseRead(
    const int hFile);

SPSSDIO_API int __stdcall spssCloseWrite(
    const int hFile);

SPSSDIO_API int __stdcall spssCommitCaseRecord(
    const int hFile);

SPSSDIO_API int __stdcall spssCommitHeader(
    const int hFile);

SPSSDIO_API int __stdcall spssConvertDate(
    const int day,
    const int month,
    const int year,
    double* spssDate);

SPSSDIO_API int __stdcall spssConvertSPSSDate(
    int* day,
    int* month,
    int* year,
    const double spssDate);

SPSSDIO_API int __stdcall spssConvertSPSSTime(
    long* day,
    int* hourh,
    int* minute,
    double* second,
    const double spssDate);

SPSSDIO_API int __stdcall spssConvertTime(
    const long day,
    const int hour,
    const int minute,
    const double second,
    double* spssTime);

SPSSDIO_API int __stdcall spssCopyDocuments(
    const int fromHandle,
    const int toHandle);

SPSSDIO_API int __stdcall spssFreeAttributes(
    char** attribNames,
    char** attribText,
    const int nAttributes);

SPSSDIO_API int __stdcall spssFreeDateVariables(
    long* dateInfo);

SPSSDIO_API int __stdcall spssFreeMultRespDefs(
    char* mrespDefs);

SPSSDIO_API int __stdcall spssFreeMultRespDefStruct(
    spssMultRespDef* pSet);

SPSSDIO_API int __stdcall spssFreeVarCValueLabels(
    char* * values,
    char* * labels,
    const int numLabels);

SPSSDIO_API int __stdcall spssFreeVariableSets(
    char* varSets);

SPSSDIO_API int __stdcall spssFreeVarNames(
    char* * varNames,
    int* varTypes,
    const int numVars);

SPSSDIO_API int __stdcall spssFreeVarNValueLabels(
    double* values,
    char* * labels,
    const int numLabels);

SPSSDIO_API int __stdcall spssGetCaseSize(
    const int hFile,
    long* caseSize);

SPSSDIO_API int __stdcall spssGetCaseWeightVar(
    const int hFile,
    char* varName);

SPSSDIO_API int __stdcall spssGetCompression(
    const int hFile,
    int* compSwitch);

SPSSDIO_API int __stdcall spssGetDateVariables(
    const int hFile,
    int* numofElements,
    long* * dateInfo);

SPSSDIO_API int __stdcall spssGetDEWFirst(
    const int hFile,
    void* pData,
    const long maxData,
    long* nData);

SPSSDIO_API int __stdcall spssGetDEWGUID(
    const int hFile,
    char* asciiGUID);

SPSSDIO_API int __stdcall spssGetDEWInfo(
    const int hFile,
    long* pLength,
    long* pHashTotal);

SPSSDIO_API int __stdcall spssGetDEWNext(
    const int hFile,
    void* pData,
    const long maxData,
    long* nData);

SPSSDIO_API int __stdcall spssGetEstimatedNofCases(
    const int hFile,
    long* caseCount);

SPSSDIO_API int __stdcall spssGetFileAttributes(
    const int hFile,
    char*** attribNames,
    char*** attribText,
    int* nAttributes);

SPSSDIO_API int __stdcall spssGetFileCodePage(
    const int hFile,
    int* nCodePage);

SPSSDIO_API int __stdcall spssGetFileEncoding(
    const int hFile,
    char* pszEncoding);

SPSSDIO_API int __stdcall spssGetIdString(
    const int hFile,
    char* id);

SPSSDIO_API int __stdcall spssGetInterfaceEncoding();

SPSSDIO_API int __stdcall spssGetMultRespCount(
    const int hFile,
    int* nSets);

SPSSDIO_API int __stdcall spssGetMultRespDefByIndex(
    const int hFile,
    const int iSet,
    spssMultRespDef** ppSet);

SPSSDIO_API int __stdcall spssGetMultRespDefs(
    const int hFile,
    char** mrespDefs);

SPSSDIO_API int __stdcall spssGetMultRespDefsEx(
    const int hFile,
    char** mrespDefs);

SPSSDIO_API int __stdcall spssGetNumberofCases(
    const int hFile,
    long* caseCount);

SPSSDIO_API int __stdcall spssGetNumberofVariables(
    const int hFile,
    int* numVars);

SPSSDIO_API int __stdcall spssGetReleaseInfo(
    const int hFile,
    int relInfo[]);

SPSSDIO_API int __stdcall spssGetSystemString(
    const int hFile,
    char* sysName);

SPSSDIO_API int __stdcall spssGetTextInfo(
    const int hFile,
    char* textInfo);

SPSSDIO_API int __stdcall spssGetTimeStamp(
    const int hFile,
    char* fileDate,
    char* fileTime);

SPSSDIO_API int __stdcall spssGetValueChar(
    const int hFile,
    const double varHandle,
    char* value,
    const int valueSize);

SPSSDIO_API int __stdcall spssGetValueNumeric(
    const int hFile,
    const double varHandle,
    double* value);

SPSSDIO_API int __stdcall spssGetVarAlignment(
    const int hFile,
    const char* varName,
    int* alignment);

SPSSDIO_API int __stdcall spssGetVarAttributes(
    const int hFile,
    const char* varName,
    char*** attribNames,
    char*** attribText,
    int* nAttributes);

SPSSDIO_API int __stdcall spssGetVarCMissingValues(
    const int hFile,
    const char* varName,
    int * missingFormat,
    char* missingVal1,
    char* missingVal2,
    char* missingVal3);

SPSSDIO_API int __stdcall spssGetVarColumnWidth(
    const int hFile,
    const char* varName,
    int* columnWidth);

SPSSDIO_API int __stdcall spssGetVarCompatName(
    const int hFile,
    const char* longName,
    char* shortName);

SPSSDIO_API int __stdcall spssGetVarCValueLabel(
    const int hFile,
    const char* varName,
    const char* value,
    char* label);

SPSSDIO_API int __stdcall spssGetVarCValueLabelLong(
    const int hFile,
    const char* varName,
    const char* value,
    char* labelBuff,
    const int lenBuff,
    int* lenLabel);

SPSSDIO_API int __stdcall spssGetVarCValueLabels(
    const int hFile,
    const char* varName,
    char* * * values,
    char* * * labels,
    int* numofLabels);

SPSSDIO_API int __stdcall spssGetVarHandle(
    const int hFile,
    const char* varName,
    double* varHandle);

SPSSDIO_API int __stdcall spssGetVariableSets(
    const int hFile,
    char* * varSets);

SPSSDIO_API int __stdcall spssGetVarInfo(
    const int hFile,
    const int iVar,
    char* varName,
    int* varType);

SPSSDIO_API int __stdcall spssGetVarLabel(
    const int hFile,
    const char* varName,
    char* varLabel);

SPSSDIO_API int __stdcall spssGetVarLabelLong(
    const int hFile,
    const char* varName,
    char* labelBuff,
    const int lenBuff,
    int* lenLabel);

SPSSDIO_API int __stdcall spssGetVarMeasureLevel(
    const int hFile,
    const char* varName,
    int* measureLevel);

SPSSDIO_API int __stdcall spssGetVarRole(
    const int hFile,
    const char* varName,
    int* varRole);

SPSSDIO_API int __stdcall spssGetVarNames(
    const int hFile,
    int* numVars,
    char* * * varNames,
    int* * varTypes);

SPSSDIO_API int __stdcall spssGetVarNMissingValues(
    const int hFile,
    const char* varName,
    int* missingFormat,
    double* missingVal1,
    double* missingVal2,
    double* missingVal3);

SPSSDIO_API int __stdcall spssGetVarNValueLabel(
    const int hFile,
    const char* varName,
    const double value,
    char* label);

SPSSDIO_API int __stdcall spssGetVarNValueLabelLong(
    const int hFile,
    const char* varName,
    const double value,
    char* labelBuff,
    const int lenBuff,
    int* lenLabel);

SPSSDIO_API int __stdcall spssGetVarNValueLabels(
    const int hFile,
    const char* varName,
    double* * values,
    char* * * labels,
    int* numofLabels);

SPSSDIO_API int __stdcall spssGetVarPrintFormat(
    const int hFile,
    const char* varName,
    int* printType,
    int* printDec,
    int* printWidth);

SPSSDIO_API int __stdcall spssGetVarWriteFormat(
    const int hFile,
    const char* varName,
    int* writeType,
    int* writeDec,
    int* writeWidth);

SPSSDIO_API void __stdcall spssHostSysmisVal(
    double* missVal);

SPSSDIO_API int __stdcall spssIsCompatibleEncoding(
    const int hFile,
    int* bCompatible);

SPSSDIO_API void __stdcall spssLowHighVal(
    double* lowest,
    double* highest);

SPSSDIO_API int __stdcall spssOpenAppend(
    const char* fileName,
    int* hFile);

SPSSDIO_API int __stdcall spssOpenAppendU8(
    const char* fileName,
    int* hFile);

SPSSDIO_API int __stdcall spssOpenRead(
    const char* fileName,
    int* hFile);

SPSSDIO_API int __stdcall spssOpenReadU8(
    const char* fileName,
    int* hFile);

SPSSDIO_API int __stdcall spssOpenWrite(
    const char* fileName,
    int* hFile);

SPSSDIO_API int __stdcall spssOpenWriteU8(
    const char* fileName,
    int* hFile);

SPSSDIO_API int __stdcall spssOpenWriteCopy(
    const char* fileName,
    const char* dictFileName,
    int* hFile);

SPSSDIO_API int __stdcall spssOpenWriteCopyU8(
    const char* fileName,
    const char* dictFileName,
    int* hFile);

SPSSDIO_API int __stdcall spssQueryType7(
    const int hFile,
    const int subType,
    int* bFound);

SPSSDIO_API int __stdcall spssReadCaseRecord(
    const int hFile);

SPSSDIO_API int __stdcall spssSeekNextCase(
    const int hFile,
    const long caseNumber);

SPSSDIO_API int __stdcall spssSetCaseWeightVar(
    const int hFile,
    const char* varName);

SPSSDIO_API int __stdcall spssSetCompression(
    const int hFile,
    const int compSwitch);

SPSSDIO_API int __stdcall spssSetDateVariables(
    const int hFile,
    const int numofElements,
    const long* dateInfo);

SPSSDIO_API int __stdcall spssSetDEWFirst(
    const int hFile,
    const void* pData,
    const long nBytes);

SPSSDIO_API int __stdcall spssSetDEWGUID(
    const int hFile,
    const char* asciiGUID);

SPSSDIO_API int __stdcall spssSetDEWNext(
    const int hFile,
    const void* pData,
    const long nBytes);

SPSSDIO_API int __stdcall spssSetFileAttributes(
    const int hFile,
    const char** attribNames,
    const char** attribText,
    const int nAttributes);

SPSSDIO_API int __stdcall spssSetIdString(
    const int hFile,
    const char* id);

SPSSDIO_API int __stdcall spssSetInterfaceEncoding(
    const int iEncoding);

SPSSDIO_API const char* __stdcall spssSetLocale(
    const int iCategory,
    const char* pszLocale);

SPSSDIO_API int __stdcall spssSetMultRespDefs(
    const int hFile,
    const char* mrespDefs);

SPSSDIO_API int __stdcall spssSetTempDir(
    const char* dirName);

SPSSDIO_API int __stdcall spssSetTextInfo(
    const int hFile,
    const char* textInfo);

SPSSDIO_API int __stdcall spssSetValueChar(
    const int hFile,
    const double varHandle,
    const char* value);

SPSSDIO_API int __stdcall spssSetValueNumeric(
    const int hFile,
    const double varHandle,
    const double value);

SPSSDIO_API int __stdcall spssSetVarAlignment(
    const int hFile,
    const char* varName,
    const int alignment);

SPSSDIO_API int __stdcall spssSetVarAttributes(
    const int hFile,
    const char* varName,
    const char** attribNames,
    const char** attribText,
    const int nAttributes);

SPSSDIO_API int __stdcall spssSetVarCMissingValues(
    const int hFile,
    const char* varName,
    const int missingFormat,
    const char* missingVal1,
    const char* missingVal2,
    const char* missingVal3);

SPSSDIO_API int __stdcall spssSetVarColumnWidth(
    const int hFile,
    const char* varName,
    const int columnWidth);

SPSSDIO_API int __stdcall spssSetVarCValueLabel(
    const int hFile,
    const char* varName,
    const char* value,
    const char* label);

SPSSDIO_API int __stdcall spssSetVarCValueLabels(
    const int hFile,
    const char* * varNames,
    const int numofVars,
    const char* * values,
    const char* * labels,
    const int numofLabels);

SPSSDIO_API int __stdcall spssSetVariableSets(
    const int hFile,
    const char* varSets);

SPSSDIO_API int __stdcall spssSetVarLabel(
    const int hFile,
    const char* varName,
    const char* varLabel);

SPSSDIO_API int __stdcall spssSetVarMeasureLevel(
    const int hFile,
    const char* varName,
    const int measureLevel);

SPSSDIO_API int __stdcall spssSetVarRole(
    const int hFile,
    const char* varName,
    const int VarRole);

SPSSDIO_API int __stdcall spssSetVarName(
    const int hFile,
    const char* varName,
    const int varType);

SPSSDIO_API int __stdcall spssSetVarNMissingValues(
    const int hFile,
    const char* varName,
    const int missingFormat,
    const double missingVal1,
    const double missingVal2,
    const double missingVal3);

SPSSDIO_API int __stdcall spssSetVarNValueLabel(
    const int hFile,
    const char* varName,
    const double value,
    const char* label);

SPSSDIO_API int __stdcall spssSetVarNValueLabels(
    const int hFile,
    const char* * varNames,
    const int numofVars,
    const double* values,
    const char* * labels,
    const int numofLabels);

SPSSDIO_API int __stdcall spssSetVarPrintFormat(
    const int hFile,
    const char* varName,
    const int printType,
    const int printDec,
    const int printWidth);

SPSSDIO_API int __stdcall spssSetVarWriteFormat(
    const int hFile,
    const char* varName,
    const int writeType,
    const int writeDec,
    const int writeWidth);

SPSSDIO_API double __stdcall spssSysmisVal(
    void);

SPSSDIO_API int __stdcall spssValidateVarname(
    const char* varName);

SPSSDIO_API int __stdcall spssWholeCaseIn(
    const int hFile,
    void* caseRec);

SPSSDIO_API int __stdcall spssWholeCaseOut(
    const int hFile,
    const void* caseRec);

#ifdef __cplusplus
    }
#endif

#endif
