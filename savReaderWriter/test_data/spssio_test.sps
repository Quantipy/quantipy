******************************************************************************.
* Title: Spss syntax to create test date for SavReaderWriter.py.
* Author: Albert-Jan Roskam.
* Date: december 2012.
******************************************************************************.

FILE HANDLE path /NAME = "%temp%".
OMS /SELECT ALL /DESTINATION FORMAT=PDF OUTFILE='path/spssio_test.pdf'.

* make some test data.
SET SEED = 4321.

DATA LIST LIST /ID Age Region Income1 Income2 Income3.
BEGIN DATA
1 27 1 35500 42700 40250
2 34 2 72300 75420 81000
3 50 1 85400 82900 84350
4 27 1 35500 42700 40250
5 27 1 35500 42700 40250
6 34 2 72300 75420 81000
7 34 2 72300 75420 81000
8 34 2 72300 75420 81000
9 50 1 85400 82900 84350
END DATA.

COMPUTE AvgIncome=MEAN(Income1, Income2, Income3).
COMPUTE MaxIncome=MAX(Income1, Income2, Income3).
COMPUTE AGE2 = AGE.
COMPUTE AGE3 = AGE.
DO REPEAT #X = SEX V1 TO V3.
COMPUTE #X = RND(UNIFORM(1)). 
END REPEAT PRINT.
IF ( MOD($casenum, 2) EQ 0 ) income3 = $sysmis.
COMPUTE someDate = DATE.DMY(21, 12, 2012).  /* dooms day ;-).

**********.
* Below are the various Spss dictionary items, in the order in which they
* appear in the Header class and the SavWriter constructor.

**********.
STRING aShortStringVar (a1) aLongStringVar (A100).
COMPUTE aShortStringVar = "x".
IF ( MOD($casenum, 2) EQ 0 ) aShortStringVar = "y".
COMPUTE aLongStringVar = "qwertyuiopasdfghjklzxcvbnm,./".

**********.
VALUE LABELS age 27 '27 y.o. ' 34 '34 y.o.' 50 '50 y.o.' 
  / aShortStringVar 'x' 'someValue label'.

**********.

VARIABLE LABEL age 'How old are you?' 
  / region 'What region do you live' 
  / aShortStringVar 'Some mysterious short stringVar'
  / aLongStringVar  'Some mysterious long stringVar'.

**********.
FORMATS id (N6) age (F3)  someDate (ADATE40).

**********.
MISSING VALUES income1 (LO THRU -1).        /* range (lower, upper).
MISSING VALUES income2 (LO THRU -1, 999). /* range + value.
MISSING VALUES income3 (999, 888, 777).      /* thee values.
MISSING VALUES age (0 THRU 18).
MISSING VALUES aShortStringVar ("x", "y").

**********.
VARIABLE LEVEL sex (NOMINAL) income1 (SCALE). 

**********.
VARIABLE WIDTH ID Age Region (10) Income1 Income2 (14) Income3 (15) someDate (13).

**********.
VARIABLE ALIGNMENT ID Age Region (left) Income1 Income2 (right) Income3 (center).

**********.
* Variable Sets are only visible in the user interface dialog boxes and the Data Editor. They cannot be used in syntax .
* I added two sets (1) incomes -->  Income1 Income2 Income3 (2) ages --> age, age2. 

**********.
VARIABLE ROLE /INPUT age /TARGET income1 income2 income3 /PARTITION region.  

**********.
VARIABLE ATTRIBUTE
  VARIABLES = AvgIncome
    ATTRIBUTE = Formula('mean(Income1, Income2, Income3)') /
  VARIABLES = MaxIncome
    ATTRIBUTE = Formula('max(Income1, Income2, Income3)') /
  VARIABLES = AvgIncome MaxIncome
    ATTRIBUTE = DerivedFrom[1]('Income1')
                         DerivedFrom[2]('Income2')
                         DerivedFrom[3]('Income3') /
  VARIABLES = ALL ATTRIBUTE=Notes('').
DISPLAY ATTRIBUTES.

**********.
DATAFILE ATTRIBUTE ATTRIBUTE=VersionNumber ('1').

**********.
FILE LABEL "This is a file label".

COMPUTE weightVar = 1.
WEIGHT BY weightVar. 

**********.
** Multiple response sets.
* category groups.
MRSETS
/MCGROUP NAME=$incomes
LABEL='three kinds of income'
VARIABLES=Income1 Income2 Income3
/DISPLAY NAME=[$incomes].

* dichotomy groups.
MRSETS 
  /MDGROUP NAME=$V CATEGORYLABELS=VARLABELS VARIABLES=v1 v2 v3 VALUE=1 
  /DISPLAY NAME=[$V]. 

CTABLES 
  /VLABELS VARIABLES=sex $V DISPLAY=DEFAULT 
  /TABLE $V [COUNT F40.0, COLPCT.COUNT PCT40.1, COLPCT.TOTALN PCT40.1] BY sex 
  /CATEGORIES VARIABLES=sex ORDER=A KEY=VALUE EMPTY=EXCLUDE 
  /CATEGORIES VARIABLES=$V  EMPTY=INCLUDE TOTAL=YES POSITION=AFTER. 

** category groups.
MRSETS
  /MCGROUP NAME=$incomes LABEL='three kinds of income' VARIABLES=Income1
  Income2 Income3 Age AGE2 AGE3
  /MCGROUP NAME=$ages LABEL='the ages' VARIABLES=Age AGE2 AGE3
  /DISPLAY NAME=[$incomes $ages].

CTABLES
  /VLABELS VARIABLES=SEX $ages DISPLAY=DEFAULT
  /TABLE $ages [COUNT F40.0] BY SEX
  /CATEGORIES VARIABLES=SEX $ages ORDER=A KEY=VALUE EMPTY=EXCLUDE.

**********.
DATE YEAR 2012 QUARTER 4 4 MONTH 12 12.

* The following new variables are being created:.
*  Name        Label.
*  YEAR_       YEAR, not periodic.
*  QUARTER_    QUARTER, period 4.
*  MONTH_      MONTH, period 12.
*  DATE_       Date.  Format:  "MMM YYYY".

**********.
* Add document.
ADD DOCUMENT "This is a an'add document' entry".
DISPLAY DOCUMENTS.

**********.
DISPLAY DICTIONARY.

SAVE OUTFILE = "path/spssio_test.sav" / COMPRESSED.
OUTPUT SAVE OUTFILE = "path/spssio_test.spv".

OMSEND.
