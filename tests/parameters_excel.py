
# BASIC 
# -----------------------------------------------------------------------------
PATH_BASIC = './tests/basic.xlsx'
XKEYS_BASIC = ['q2', 'q2b', 'q3', 'q4',
               ['q5_1', 'q5_2', 'q5_3', 'q5_4', 'q5_5', 'q5_6'],
               'q8', 'q9']
YKEYS_BASIC = ['@', 'gender', 'locality']
VIEWS_BASIC = ['cbase', 'counts']
OPENS_BASIC = ['q8a', 'q9a']
CELLS_BASIC = 'counts'
WEIGHT_BASIC = None
# -----------------------------------------------------------------------------

# COMPLEX 1
# -----------------------------------------------------------------------------
PATH_COMPLEX_0 = './tests/complex_0.xlsx'

XKEYS_COMPLEX = ['q5_1', 'q4', 'gender', 'Wave']
YKEYS_COMPLEX = ['@', 'q4 > gender', 'q4 > gender > Wave', 'q5_1']
VIEWS_COMPLEX = ['cbase', 'cbase_gross', 'ebase', 'counts', 'c%', 'r%',
                 'counts_sum', 'c%_sum']
OPENS_COMPLEX = ['RecordNo', 'gender', 'age', 'q8', 'q8a', 'q9', 'q9a']
WEIGHT_COMPLEX = 'weight_a'

VIEWS_COMPLEX_MAIN = ["x|f|x:|||cbase",
                      "x|f|x:||weight_a|cbase",
                      "x|f|x:|||cbase_gross",
                      "x|f|x:||weight_a|cbase_gross",
                      "x|f|x:|||ebase",
                      "x|f|x:||weight_a|ebase",
                      (
                          "x|f|:||weight_a|counts",
                          "x|f|:|y|weight_a|c%",
                          "x|f|:|x|weight_a|r%",
                          "x|t.props.Dim.80+@|:||weight_a|test"
                      ),
                      (
                          "x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:||weight_a|NPS",
                          "x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|y|weight_a|NPS",
                          "x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|x|weight_a|NPS",
                          "x|t.props.Dim.80+@|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:||weight_a|test"
                      ),
                      (
                          "x|f.c:f|x[{4,5}-{1,2}]:||weight_a|NPSonly",
                          "x|f.c:f|x[{4,5}-{1,2}]:|y|weight_a|NPSonly",
                          "x|f.c:f|x[{4,5}-{1,2}]:|x|weight_a|NPSonly",
                          "x|t.props.Dim.80+@|x[{4,5}-{1,2}]:||weight_a|test"
                      ),
                      (
                          "x|f|x[{1,2,3}]:||weight_a|No",
                          "x|f|x[{1,2,3}]:|y|weight_a|No",
                          "x|f|x[{1,2,3}]:|x|weight_a|No",
                          "x|t.props.Dim.80+@|x[{1,2,3}]:||weight_a|test"
                      ),
                      (
                          "x|f|x[{4,5,97}]:||weight_a|Yes",
                          "x|f|x[{4,5,97}]:|y|weight_a|Yes",
                          "x|f|x[{4,5,97}]:|x|weight_a|Yes",
                          "x|t.props.Dim.80+@|x[{4,5,97}]:||weight_a|test"
                      ),
                      (
                          "x|d.mean|x:||weight_a|stat",
                          "x|t.means.Dim.80+@|x:||weight_a|test"
                      ),
                      "x|d.stddev|x:||weight_a|stat",
                      "x|d.median|x:||weight_a|stat",
                      "x|d.var|x:||weight_a|stat",
                      "x|d.varcoeff|x:||weight_a|stat",
                      "x|d.sem|x:||weight_a|stat",
                      "x|d.lower_q|x:||weight_a|stat",
                      "x|d.upper_q|x:||weight_a|stat",
                      (
                          "x|f.c:f|x:||weight_a|counts_sum",
                          "x|f.c:f|x:|y|weight_a|c%_sum"
                      )]

VIEWS_COMPLEX_WAVE = ["x|f|x:|||cbase",
                      "x|f|x:||weight_a|cbase",
                      "x|f|x:|||cbase_gross",
                      "x|f|x:||weight_a|cbase_gross",
                      "x|f|x:|||ebase",
                      "x|f|x:||weight_a|ebase",
                      (
                          "x|f|x[{1,2}+],x[{4,5}+]*:||weight_a|BLOCK",
                          "x|f|x[{1,2}+],x[{4,5}+]*:|y|weight_a|BLOCK",
                          "x|f|x[{1,2}+],x[{4,5}+]*:|x|weight_a|BLOCK",
                          "x|t.props.Dim.80+@|x[{1,2}+],x[{4,5}+]*:||weight_a|test"
                      ),
                      (
                          "x|d.mean|x:||weight_a|stat",
                          "x|t.means.Dim.80+@|x:||weight_a|test"
                      ),
                      "x|d.stddev|x:||weight_a|stat",
                      "x|d.median|x:||weight_a|stat",
                      "x|d.var|x:||weight_a|stat",
                      "x|d.varcoeff|x:||weight_a|stat",
                      "x|d.sem|x:||weight_a|stat",
                      "x|d.lower_q|x:||weight_a|stat",
                      "x|d.upper_q|x:||weight_a|stat",
                      (
                          "x|f.c:f|x:||weight_a|counts_sum",
                          "x|f.c:f|x:|y|weight_a|c%_sum"
                      )]
# -----------------------------------------------------------------------------
