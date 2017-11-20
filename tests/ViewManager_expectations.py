
# cell_items: c/p/cp
# basics:     b
# nets:       n
# stats:      s
# tests:      t
# weights:    w
# bases:      auto/both/wgt/unwgt

EXPECT = {
        ########################### test_basics ##########################
        # counts, basics, unweighted
        'c_b':[
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|:|||counts',
            'x|f.c:f|x++:|||counts_cumsum',
            'x|f.c:f|x:|||counts_sum'],
        # percentages, basics, unweighted
        'p_b':[
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|:|y||c%',
            'x|f.c:f|x++:|y||c%_cumsum',
            'x|f.c:f|x:|y||c%_sum'],
        # counts + percentages, basics, unweighted
        'cp_b':[
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            ('x|f|:|||counts', 'x|f|:|y||c%'),
            ('x|f.c:f|x++:|||counts_cumsum', 'x|f.c:f|x++:|y||c%_cumsum'),
            ('x|f.c:f|x:|||counts_sum', 'x|f.c:f|x:|y||c%_sum')],
        # counts, basics, weighted, base auto
        'c_b_w_auto':[
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            'x|f|:||weight_a|counts',
            'x|f.c:f|x++:||weight_a|counts_cumsum',
            'x|f.c:f|x:||weight_a|counts_sum'],
        # percentage, basics, weighted, base auto
        'p_b_w_auto':[
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            'x|f|:|y|weight_a|c%',
            'x|f.c:f|x++:|y|weight_a|c%_cumsum',
            'x|f.c:f|x:|y|weight_a|c%_sum'],
        # counts + percentage, basics, weighted, base auto
        'cp_b_w_auto':[
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:||weight_a|counts', 'x|f|:|y|weight_a|c%'),
            ('x|f.c:f|x++:||weight_a|counts_cumsum', 'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
            ('x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:|y|weight_a|c%_sum')],
        # counts, basics, weighted, base both
        'c_b_w_both':[
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            'x|f|:||weight_a|counts',
            'x|f.c:f|x++:||weight_a|counts_cumsum',
            'x|f.c:f|x:||weight_a|counts_sum'],
        # percentage, basics, weighted, base both
        'p_b_w_both':[
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            'x|f|:|y|weight_a|c%',
            'x|f.c:f|x++:|y|weight_a|c%_cumsum',
            'x|f.c:f|x:|y|weight_a|c%_sum'],
        # counts + percentage, basics, weighted, base both
        'cp_b_w_both':[
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:||weight_a|counts', 'x|f|:|y|weight_a|c%'),
            ('x|f.c:f|x++:||weight_a|counts_cumsum', 'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
            ('x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:|y|weight_a|c%_sum')],

        ########################## test_complex ##########################
        # counts + percentage, basics, nets, weighted, base both
        'cp_b_n_w_both':[
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:||weight_a|counts', 'x|f|:|y|weight_a|c%'),
            ('x|f.c:f|x++:||weight_a|counts_cumsum', 'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
            ('x|f|x[{1,2,3}]:||weight_a|net', 'x|f|x[{1,2,3}]:|y|weight_a|net'),
            ('x|f|x[{1,2}+]*:||weight_a|net', 'x|f|x[{1,2}+]*:|y|weight_a|net'),
            ('x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:|y|weight_a|c%_sum')],
        # percentage, basics, nets, weighted, base both
        'p_b_n_w_both':[
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            'x|f|:|y|weight_a|c%',
            'x|f.c:f|x++:|y|weight_a|c%_cumsum',
            'x|f|x[{1,2,3}]:|y|weight_a|net',
            'x|f|x[{1,2}+]*:|y|weight_a|net',
            'x|f.c:f|x:|y|weight_a|c%_sum'],
        # counts + percentage, basics, stats, weighted, base both
        'cp_b_s_w_both': [
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:||weight_a|counts', 'x|f|:|y|weight_a|c%'),
            ('x|f.c:f|x++:||weight_a|counts_cumsum', 'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
            ('x|d.mean|x:||weight_a|stat',),
            'x|d.stddev|x:||weight_a|stat',
            'x|d.mean|x[{100,50,0}]:||weight_a|stat',
            ('x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:|y|weight_a|c%_sum')],
        # percentage, basics, stats, weighted, base both
        'p_b_s_w_both': [
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            'x|f|:|y|weight_a|c%',
            'x|f.c:f|x++:|y|weight_a|c%_cumsum',
            'x|d.mean|x:||weight_a|stat',
            'x|d.stddev|x:||weight_a|stat',
            'x|d.mean|x[{100,50,0}]:||weight_a|stat',
            'x|f.c:f|x:|y|weight_a|c%_sum'],
        # counts + percentage, basics, tests, weighted, base both
        'cp_b_t_w_both': [
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:||weight_a|counts', 'x|f|:|y|weight_a|c%', 'x|t.props.Dim.05|:||weight_a|significance'),
            ('x|f.c:f|x++:||weight_a|counts_cumsum', 'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
            ('x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:|y|weight_a|c%_sum')],
        # percentage, basics, tests, weighted, base both
        'p_b_t_w_both': [
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:|y|weight_a|c%', 'x|t.props.Dim.05|:||weight_a|significance'),
            'x|f.c:f|x++:|y|weight_a|c%_cumsum',
            'x|f.c:f|x:|y|weight_a|c%_sum'],
        # counts + percentage, basics, nets, stats, weighted, base both
        'cp_b_n_s_w_both': [
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:||weight_a|counts', 'x|f|:|y|weight_a|c%'),
            ('x|f.c:f|x++:||weight_a|counts_cumsum', 'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
            ('x|f|x[{1,2,3}]:||weight_a|net', 'x|f|x[{1,2,3}]:|y|weight_a|net'),
            ('x|f|x[{1,2}+]*:||weight_a|net', 'x|f|x[{1,2}+]*:|y|weight_a|net'),
            ('x|d.mean|x:||weight_a|stat',),
            'x|d.stddev|x:||weight_a|stat',
            'x|d.mean|x[{100,50,0}]:||weight_a|stat',
            ('x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:|y|weight_a|c%_sum')],
        # percentage, basics, nets, stats, weighted, base both
        'p_b_n_s_w_both': [
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            'x|f|:|y|weight_a|c%',
            'x|f.c:f|x++:|y|weight_a|c%_cumsum',
            'x|f|x[{1,2,3}]:|y|weight_a|net',
            'x|f|x[{1,2}+]*:|y|weight_a|net',
            'x|d.mean|x:||weight_a|stat',
            'x|d.stddev|x:||weight_a|stat',
            'x|d.mean|x[{100,50,0}]:||weight_a|stat',
            'x|f.c:f|x:|y|weight_a|c%_sum'],
        # counts + percentage, basics, nets, tests, weighted, base both
        'cp_b_n_t_w_both':[
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:||weight_a|counts', 'x|f|:|y|weight_a|c%', 'x|t.props.Dim.05|:||weight_a|significance'),
            ('x|f.c:f|x++:||weight_a|counts_cumsum', 'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
            ('x|f|x[{1,2,3}]:||weight_a|net', 'x|f|x[{1,2,3}]:|y|weight_a|net', 'x|t.props.Dim.05|x[{1,2,3}]:||weight_a|significance'),
            ('x|f|x[{1,2}+]*:||weight_a|net', 'x|f|x[{1,2}+]*:|y|weight_a|net', 'x|t.props.Dim.05|x[{1,2}+]*:||weight_a|significance'),
            ('x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:|y|weight_a|c%_sum')],
        # percentage, basics, nets, tests, weighted, base both
        'p_b_n_t_w_both':[
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:|y|weight_a|c%', 'x|t.props.Dim.05|:||weight_a|significance'),
            ('x|f|x[{1,2,3}]:|y|weight_a|net', 'x|t.props.Dim.05|x[{1,2,3}]:||weight_a|significance'),
            ('x|f|x[{1,2}+]*:|y|weight_a|net', 'x|t.props.Dim.05|x[{1,2}+]*:||weight_a|significance'),
            'x|f.c:f|x++:|y|weight_a|c%_cumsum',
            'x|f.c:f|x:|y|weight_a|c%_sum'],
        # counts + percentage, basics, stats, tests, weighted, base both
        'cp_b_s_t_w_both': [
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:||weight_a|counts', 'x|f|:|y|weight_a|c%', 'x|t.props.Dim.05|:||weight_a|significance'),
            ('x|f.c:f|x++:||weight_a|counts_cumsum', 'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
            ('x|d.mean|x:||weight_a|stat', 'x|t.means.Dim.05|x:||weight_a|significance'),
            ('x|d.mean|x[{100,50,0}]:||weight_a|stat', 'x|t.means.Dim.05|x[{100,50,0}]:||weight_a|significance'),
            'x|d.stddev|x:||weight_a|stat',
            ('x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:|y|weight_a|c%_sum')],
        # percentage, basics, stats, tests, weighted, base both
        'p_b_s_t_w_both': [
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:|y|weight_a|c%', 'x|t.props.Dim.05|:||weight_a|significance'),
            ('x|d.mean|x:||weight_a|stat', 'x|t.means.Dim.05|x:||weight_a|significance'),
            ('x|d.mean|x[{100,50,0}]:||weight_a|stat', 'x|t.means.Dim.05|x[{100,50,0}]:||weight_a|significance'),
            'x|d.stddev|x:||weight_a|stat',
            'x|f.c:f|x++:|y|weight_a|c%_cumsum',
            'x|f.c:f|x:|y|weight_a|c%_sum'],
        # counts + percentage, basics, nets, stats, tests, weighted, base both
        'cp_b_n_s_t_w_both': [
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:||weight_a|counts', 'x|f|:|y|weight_a|c%', 'x|t.props.Dim.05|:||weight_a|significance'),
            ('x|f.c:f|x++:||weight_a|counts_cumsum', 'x|f.c:f|x++:|y|weight_a|c%_cumsum'),
            ('x|f|x[{1,2,3}]:||weight_a|net', 'x|f|x[{1,2,3}]:|y|weight_a|net', 'x|t.props.Dim.05|x[{1,2,3}]:||weight_a|significance'),
            ('x|f|x[{1,2}+]*:||weight_a|net', 'x|f|x[{1,2}+]*:|y|weight_a|net', 'x|t.props.Dim.05|x[{1,2}+]*:||weight_a|significance'),
            ('x|d.mean|x:||weight_a|stat', 'x|t.means.Dim.05|x:||weight_a|significance'),
            ('x|d.mean|x[{100,50,0}]:||weight_a|stat', 'x|t.means.Dim.05|x[{100,50,0}]:||weight_a|significance'),
            'x|d.stddev|x:||weight_a|stat',
            ('x|f.c:f|x:||weight_a|counts_sum', 'x|f.c:f|x:|y|weight_a|c%_sum')],
        # percentage, basics,, nets stats, tests, weighted, base both
        'p_b_n_s_t_w_both': [
            'x|f|x:|||cbase_gross',
            'x|f|x:|||cbase',
            'x|f|x:||weight_a|cbase_gross',
            'x|f|x:||weight_a|cbase',
            'x|f|x:||weight_a|ebase',
            ('x|f|:|y|weight_a|c%', 'x|t.props.Dim.05|:||weight_a|significance'),
            ('x|f|x[{1,2,3}]:|y|weight_a|net', 'x|t.props.Dim.05|x[{1,2,3}]:||weight_a|significance'),
            ('x|f|x[{1,2}+]*:|y|weight_a|net', 'x|t.props.Dim.05|x[{1,2}+]*:||weight_a|significance'),
            ('x|d.mean|x:||weight_a|stat', 'x|t.means.Dim.05|x:||weight_a|significance'),
            ('x|d.mean|x[{100,50,0}]:||weight_a|stat', 'x|t.means.Dim.05|x[{100,50,0}]:||weight_a|significance'),
            'x|d.stddev|x:||weight_a|stat',
            'x|f.c:f|x++:|y|weight_a|c%_cumsum',
            'x|f.c:f|x:|y|weight_a|c%_sum']
            }
