
# BASIC 
# -----------------------------------------------------------------------------
PATH_BASIC = './tests/basic.xlsx'
XKEYS_BASIC = ('q2', 'q2b', 'q3', 'q4',
               ('q5_1', 'q5_2', 'q5_3', 'q5_4', 'q5_5', 'q5_6'),
               'q8', 'q9')
YKEYS_BASIC = ('@', 'gender', 'locality')
VIEWS_BASIC = ('cbase', 'counts')
OPENS_BASIC = ('q8a', 'q9a')
CELLS_BASIC = 'counts'
WEIGHT_BASIC = None
# -----------------------------------------------------------------------------

# COMPLEX 1
# -----------------------------------------------------------------------------
PATH_COMPLEX_0 = './tests/complex_0.xlsx'
PATH_COMPLEX_1 = './tests/complex_1.xlsx'
PATH_COMPLEX_2 = './tests/complex_2.xlsx'
PATH_COMPLEX_3 = './tests/complex_3.xlsx'

XKEYS_COMPLEX = ('q5_1', 'q4', 'gender', 'Wave')
YKEYS_COMPLEX = ('@', 'q4 > gender', 'q4 > gender > Wave', 'q5_1')
VIEWS_COMPLEX = ('cbase', 'cbase_gross', 'ebase', 'counts', 'c%', 'r%',
                 'counts_sum', 'c%_sum')
OPENS_COMPLEX = ('RecordNo', 'gender', 'age', 'q8', 'q8a', 'q9', 'q9a')
WEIGHT_COMPLEX = 'weight_a'

VIEWS_COMPLEX_MAIN = ('x|f|x:|||cbase',
                      'x|f|x:||weight_a|cbase',
                      'x|f|x:|||cbase_gross',
                      'x|f|x:||weight_a|cbase_gross',
                      'x|f|x:|||ebase',
                      'x|f|x:||weight_a|ebase',
                      (
                          'x|f|:||weight_a|counts',
                          'x|f|:|y|weight_a|c%',
                          'x|f|:|x|weight_a|r%',
                          'x|t.props.Dim.80+@|:||weight_a|test'
                      ),
                      (
                          'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:||weight_a|NPS',
                          'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|y|weight_a|NPS',
                          'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|x|weight_a|NPS',
                          'x|t.props.Dim.80+@|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:||weight_a|test'
                      ),
                      (
                          'x|f.c:f|x[{4,5}-{1,2}]:||weight_a|NPSonly',
                          'x|f.c:f|x[{4,5}-{1,2}]:|y|weight_a|NPSonly',
                          'x|f.c:f|x[{4,5}-{1,2}]:|x|weight_a|NPSonly',
                          'x|t.props.Dim.80+@|x[{4,5}-{1,2}]:||weight_a|test'
                      ),
                      (
                          'x|f|x[{1,2,3}]:||weight_a|No',
                          'x|f|x[{1,2,3}]:|y|weight_a|No',
                          'x|f|x[{1,2,3}]:|x|weight_a|No',
                          'x|t.props.Dim.80+@|x[{1,2,3}]:||weight_a|test'
                      ),
                      (
                          'x|f|x[{4,5,97}]:||weight_a|Yes',
                          'x|f|x[{4,5,97}]:|y|weight_a|Yes',
                          'x|f|x[{4,5,97}]:|x|weight_a|Yes',
                          'x|t.props.Dim.80+@|x[{4,5,97}]:||weight_a|test'
                      ),
                      (
                          'x|d.mean|x:||weight_a|stat',
                          'x|t.means.Dim.80+@|x:||weight_a|test'
                      ),
                      'x|d.stddev|x:||weight_a|stat',
                      'x|d.median|x:||weight_a|stat',
                      'x|d.var|x:||weight_a|stat',
                      'x|d.varcoeff|x:||weight_a|stat',
                      'x|d.sem|x:||weight_a|stat',
                      'x|d.lower_q|x:||weight_a|stat',
                      'x|d.upper_q|x:||weight_a|stat',
                      (
                          'x|f.c:f|x:||weight_a|counts_sum',
                          'x|f.c:f|x:|y|weight_a|c%_sum'
                      )) 

VIEWS_COMPLEX_WAVE = ('x|f|x:|||cbase',
                      'x|f|x:||weight_a|cbase',
                      'x|f|x:|||cbase_gross',
                      'x|f|x:||weight_a|cbase_gross',
                      'x|f|x:|||ebase',
                      'x|f|x:||weight_a|ebase',
                      (
                          'x|f|x[{1,2}+],x[{4,5}+]*:||weight_a|BLOCK',
                          'x|f|x[{1,2}+],x[{4,5}+]*:|y|weight_a|BLOCK',
                          'x|f|x[{1,2}+],x[{4,5}+]*:|x|weight_a|BLOCK',
                          'x|t.props.Dim.80+@|x[{1,2}+],x[{4,5}+]*:||weight_a|test'
                      ),
                      (
                          'x|d.mean|x:||weight_a|stat',
                          'x|t.means.Dim.80+@|x:||weight_a|test'
                      ),
                      'x|d.stddev|x:||weight_a|stat',
                      'x|d.median|x:||weight_a|stat',
                      'x|d.var|x:||weight_a|stat',
                      'x|d.varcoeff|x:||weight_a|stat',
                      'x|d.sem|x:||weight_a|stat',
                      'x|d.lower_q|x:||weight_a|stat',
                      'x|d.upper_q|x:||weight_a|stat',
                      (
                          'x|f.c:f|x:||weight_a|counts_sum',
                          'x|f.c:f|x:|y|weight_a|c%_sum'
                      ))

VIEWS_COMPLEX_ARRAY = ('x|f|x:|||cbase',
                       'x|f|x:||weight_a|cbase',
                       (
                          'x|f|:||weight_a|counts',
                          'x|f|:|y|weight_a|c%'
                       ),
                       (
                          'x|f|x[{1,2,3}]:||weight_a|No',
                          'x|f|x[{1,2,3}]:|y|weight_a|No'
                       ),
                       (
                          'x|f|x[{4,5,97}]:||weight_a|Yes',
                          'x|f|x[{4,5,97}]:|y|weight_a|Yes'
                       ),
                       (
                          'x|f.c:f|x[{4,5}-{1,2}]:||weight_a|NPSonly',
                          'x|f.c:f|x[{4,5}-{1,2}]:|y|weight_a|NPSonly'
                       ),
                       (
                           'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:||weight_a|NPS',
                           'x|f.c:f|x[{1,2}],x[{4,5}],x[{4,5}-{1,2}]:|y|weight_a|NPS'
                       ),
                       'x|d.mean|x:||weight_a|stat',
                       'x|d.stddev|x:||weight_a|stat',
                       'x|d.median|x:||weight_a|stat',
                       'x|d.var|x:||weight_a|stat',
                       'x|d.varcoeff|x:||weight_a|stat',
                       'x|d.sem|x:||weight_a|stat',
                       'x|d.lower_q|x:||weight_a|stat',
                       'x|d.upper_q|x:||weight_a|stat',
                       (
                           'x|f.c:f|x:||weight_a|counts_sum',
                           'x|f.c:f|x:|y|weight_a|c%_sum'
                       ))

VIEWS_COMPLEX_MEAN = ('x|f|x:|||cbase',
                      'x|f|x:||weight_a|cbase',
                      'x|d.mean|x:||weight_a|stat',
                      'x|d.stddev|x:||weight_a|stat',
                      'x|d.median|x:||weight_a|stat',
                      'x|d.var|x:||weight_a|stat',
                      'x|d.varcoeff|x:||weight_a|stat',
                      'x|d.sem|x:||weight_a|stat',
                      'x|d.lower_q|x:||weight_a|stat',
                      'x|d.upper_q|x:||weight_a|stat')

SHEET_PROPERTIES_1 = dict(alternate_bg=True,
                          freq_0_rep=':',
                          stat_0_rep='#',
                          y_header_height=20,
                          y_row_height=40,
                          column_width=10,
                          column_width_label=40,
                          column_width_frame=50,
                          row_height_label=15,
                          arrow_color_high='#66023C',
                          arrow_rep_high=u'\u219F',
                          arrow_color_low='#3F0F69',
                          arrow_rep_low=u'\u21A1')

VIEW_GROUPS_1 = dict(block_expanded_counts='freq',
                     block_expanded_c_pct='freq',
                     block_expanded_r_pct='freq',
                     block_expanded_propstest='freq',
                     block_net_counts='freq',
                     block_net_c_pct='freq',
                     block_net_r_pct='freq',
                     block_net_propstest='freq')

FORMATS_1 = dict(bg_color_freq='gray')

IMAGE_1 = dict(img_name='logo',
               img_url='./qplogo_invert.png',
               img_size=[110, 120],
               img_insert_x=4,
               img_insert_y=0,
               img_x_offset=3,
               img_y_offset=6)

DECIMALS_1 = 3

#italicise_level=50,
#decimals=dict(N=0, P=2, D=1),
#details=True,
#image=image,
 # -----------------------------------------------------------------------------

