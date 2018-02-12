
# BASIC 
# -----------------------------------------------------------------------------
XKEYS = ['q2', 'q2b', 'q3', 'q4', 
         ['q5_1', 'q5_2', 'q5_3', 'q5_4', 'q5_5', 'q5_6'],
         'q8', 'q9']
YKEYS = ['@', 'gender', 'locality']
VIEWS = ['cbase', 'counts']
OPENS = ['q8a', 'q9a']
CELLS = 'counts'
EXCEL = './tests/basic.xlsx'

BASIC = (XKEYS, YKEYS, VIEWS, OPENS, CELLS, EXCEL)
# -----------------------------------------------------------------------------

# COMPLEX 1
# -----------------------------------------------------------------------------
XKEYS = ['q5_1', 'q4', 'gender', 'Wave']
YKEYS = ['@', 'q4 > gender', 'q4 > gender > Wave', 'q5_1']
VIEWS = ['cbase', 'cbase_gross', 'ebase', 'counts', 'c%', 'r%', 'counts_sum', 'c%_sum']
OPENS = ['RecordNo', 'gender', 'age', 'q8', 'q8a', 'q9', 'q9a']
CELLS = 'counts'
EXCEL = './tests/basic.xlsx'

ASDASDA = (XKEYS, YKEYS, VIEWS, OPENS, CELLS, EXCEL)
