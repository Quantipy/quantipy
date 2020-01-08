    def test_props_blocknet_calc_incl_total(self):
        from operator import add, sub
        views = ['counts']
        x, y = 'q7_1', 'q8'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_a'
            )
        nets = ViewMapper()
        nets.make_template('frequency')
        nets_def = [{'Z': [1, 2, 3], 'expand': 'after',
                     'text': {'en-GB': 'some text1'}},
                    {'A': [4, 5],
                     'text': {'en-GB': 'some text2'}},
                    {'F': [6, 7, 8], 'expand': 'before',
                     'text': {'en-GB': 'some text3'}}]
        calc = {'my_calc': ('Z', sub, 'F')}
        nets.add_method(name='blocknet',
                        kwargs={'logic': nets_def,
                                'axis': 'x',
                                'complete': True,
                                'calc': calc})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=nets, weights='weight_a')

        tests = ViewMapper()
        tests.make_template('coltests', iterators={'metric': ['props']})
        tests.add_method(name='total_tests_blocks',
                         kwargs={'test_total': True})
        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=tests, weights='weight_a')

        link = self.stack['testing']['no_filter'][x][y]
        nets_view = link['x|t.props.Dim.10+@|x[{1,2,3}+],x[{4,5}],x[+{6,7,8}],x[{1,2,3}-{6,7,8}]*:||weight_a|total_tests_blocks']

        nets_result = [["['@H']", 'NONE', '[1]', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', "['@L', 1, 3, 4, 5, 96]", 'NONE', '[5]', "['@H']", 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['[4]', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['[4]', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', '[96]', 'NONE', 'NONE'],
                       ['[5]', '[5]', 'NONE', '[5]', "['@H']", "['@L', 5]", "['@L', 1, 3, 4, 5]"],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE']]
        self.assertEqual(nets_view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         nets_result)

    def test_props_changed_meta_nets_incl_total(self):
        from copy import deepcopy
        x = 'q5_1'
        y = 'q8'
        self.setup_stack(
            views=None,
            x=x,
            y=y,
            weights=None
            )
        # re-ordering meta in a random fashion to test robustness
        meta = deepcopy(self.stack['testing'].meta)
        meta = emulate_meta(meta, meta)
        values = meta['columns']['q5_1']['values']
        slicer = [98, 1, 2, 5, 4, 3, 97]
        meta['columns']['q5_1']['values'] = self.slice_values_meta(values, slicer)
        values = meta['columns']['q8']['values']
        slicer = [98, 96, 5, 1, 2, 3, 4]
        meta['columns']['q8']['values'] = self.slice_values_meta(values, slicer)
        self.stack['testing'].meta = meta
        self.stack.add_link(x='q5_1', y='q8', views=QuantipyViews(['counts']))

        nets = ViewMapper()
        nets.make_template('frequency')
        nets_def = [{'Z': [98, 97], 'expand': 'after',
                                    'text': {'en-GB': 'some text1'}},
                    {'A': [4, 1],  'expand': 'before',
                                   'text': {'en-GB': 'some text2'}}]

        nets.add_method(name='blocknet',
                        kwargs={'logic': nets_def,
                                'axis': 'x',
                                'complete': True})
        self.stack.add_link(x='q5_1', y='q8', views=nets, weights=None)

        tests = ViewMapper()
        tests.make_template('coltests')
        tests.add_method(name='tests',
                         kwargs={'level': 0.1,
                                'test_total': True})
        self.stack.add_link(x='q5_1', y='q8', views=tests, weights=None)

        link = self.stack['testing']['no_filter']['q5_1']['q8']
        nets_test_view = link['x|t.props.Dim.10+@|x[{98,97}+],x[+{4,1}]*:|||tests']
        test_resuts = [['[5, 1, 2, 3, 4]', "['@H', 2]", "['@H']", "['@H']", "['@H']", "['@H']", "['@H']"],
                       ['[5, 1, 2, 3, 4]', "['@H', 2]", "['@H']", "['@H']", "['@H']", "['@H']", "['@H']"],
                       ['NONE', '[5]', "['@H']", "['@H', 5]", "['@H']", '[5]', "['@H']"],
                       ['NONE', 'NONE', "['@L']", "['@L']", "['@L', 96, 3, 4]", 'NONE', 'NONE'],
                       ['NONE', "['@H']", '[96]', '[96]', "['@H']", '[96]', '[96, 2]'],
                       ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE', '[96, 1, 3]'],
                       ['NONE', 'NONE', 'NONE', "['@H']", "['@L', 96, 5, 1, 3, 4]", '[1]', '[1]'],
                       ['NONE', "['@H']", '[1]', "['@H']", '[96, 5, 1, 3]', 'NONE', '[96, 1]'],
                       ['NONE', "['@L', 5, 4]", "['@L']", "['@L', 4]", "['@L']", "['@L']", "['@L']"]]
        self.assertEqual(nets_test_view.dataframe.replace(np.NaN, 'NONE').values.tolist(),
                         test_resuts)

        net_agg = link['x|f|x[{98,97}+],x[+{4,1}]*:|||blocknet']
        net_agg_idx = net_agg.dataframe.index.get_level_values(1).tolist()
        net_agg_cols = net_agg.dataframe.columns.get_level_values(1).tolist()
        net_test_idx = nets_test_view.dataframe.index.get_level_values(1).tolist()
        net_test_cols = nets_test_view.dataframe.columns.get_level_values(1).tolist()

        self.assertEqual(net_agg_idx, ['Z', 98, 97, 2, 5, 4, 1, 'A', 3])
        self.assertEqual(net_agg_cols, [98, 96, 5, 1, 2, 3, 4])
        self.assertEqual(net_agg_idx, net_test_idx)
        self.assertEqual(net_agg_cols, net_test_cols)

    def test_props_test_level_low_askia_weighted(self):
        views = QuantipyViews(['counts', 'cbase'])
        x = 'q9'
        y = 'q8'
        self.setup_stack(
            views=views,
            x=x,
            y=y,
            weights='weight_a'
            )

        prop_sig = ViewMapper(
            template={
                'method': QuantipyViews().coltests,
                'kwargs': {
                    'rel_to': 'y',
                    'stack': self.stack,
                    'iterators': {
                        'level': ['low']
                        }
                    }
                })

        prop_sig.add_method(name='askia_props_test',
                            kwargs={'text': 'SIG (props, askia_low)',
                                    'mimic': 'askia'})

        self.stack.add_link(data_keys='testing', x=x, y=y,
                            views=prop_sig, weights='weight_a')

        view = self.stack['testing']['no_filter'][x][y]['x|t.props.askia.10|:|y|weight_a|askia_props_test']
        meta = self.stack['testing']['no_filter'][x][y]['x|t.props.askia.10|:|y|weight_a|askia_props_test'].meta()

        sig_result = [['[98]', '[98]', '[2, 5, 98]', '[2, 98]', '[98]', '[98]', 'NONE'],
                      ['[96, 98]', '[96, 98]', '[1, 96, 98]', '[96, 98]', '[96, 98]', 'NONE', 'NONE'],
                      ['[98]', '[98]', '[1, 2, 4, 5, 96, 98]', '[98]', '[98]', '[98]', 'NONE'],
                      ['[98]', '[98]', '[4, 5, 98]', '[98]', '[98]', '[98]', 'NONE'],
                      ['[2]', 'NONE', '[2, 4]', 'NONE', '[2]', '[2, 4]', 'NONE'],
                      ['NONE', '[1, 3, 4, 5, 96]', 'NONE', 'NONE', '[3, 4]', 'NONE', 'NONE'],
                      ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', '[2, 3, 5]', '[1, 2, 3, 4, 5]']]


        meta_agg_text = 'SIG (props, askia_low)'
        meta_siglevel = 0.10

        self.assertEqual(view.dataframe.replace(np.NaN, 'NONE').values.tolist(), sig_result)
        self.assertEqual(meta['agg']['text'], meta_agg_text)
        self.assertEqual(view.is_propstest(), meta_siglevel)


    def setup_stack(
        self,
        key='testing',
        data=io.load_csv(os.path.dirname(os.path.abspath(__file__))+'/Example Data (A).csv'),
        meta=io.load_json(os.path.dirname(os.path.abspath(__file__))+'/Example Data (A).json'),
        filters=None,
        x=None,
        y=None,
        views=None,
        weights=None):
        self.stack = Stack('Test')
        self.stack.add_data(data_key=key, data=data, meta=meta)
        self.stack.add_link(data_keys=key, x=x, y=y, views=views, weights=weights)

    def slice_values_meta(self, values, slicer):
        """
        Return a slice of the given values meta using slicer.
        """
        new_values = [
            value
            for i in slicer
            for value in values
            if value['value']==i]
        return new_values