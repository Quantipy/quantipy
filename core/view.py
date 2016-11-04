# -*- coding: utf-8 -*-
import quantipy.core.helpers.functions as helpers
from operator import add, sub, mul, div
import pandas as pd
import copy
pd.set_option('display.encoding', 'utf-8')

class View(object):
    def __init__(self, link=None, name=None, kwargs=None):
        kwargs = None if kwargs is None else kwargs.copy()
        self._kwargs = kwargs
        self.name = name
        if not link is None:
            self._link_meta(link)
        self.dataframe = pd.DataFrame()
        self._notation = None
        self.rbases = None
        self.cbases = None
        self.grp_text_map = None
        self._custom_txt = ''
        self.add_base_text = True

    def meta(self):
        """
        Get a summary on a View's meta information.

        Returns
        -------
        viewmeta: dict
            A dictionary that contains global aggregation information.
        """
        viewmeta = {
                    'agg':
                    {
                     'is_weighted': self.is_weighted(),
                     'weights': self.get_std_params()[3],
                     'method': self._method(),
                     'name': self._shortname(),
                     'fullname': self._notation,
                     'text': self.get_std_params()[4],
                     'grp_text_map': self.grp_text_map,
                     'is_block': self._is_block()
                     },
                    'x': self._x,
                    'y': self._y,
                    'shape': self.dataframe.shape
                    }
        if self.is_base():
            viewmeta['agg']['add_base_text'] = self.add_base_text
        return viewmeta

    def _link_meta(self, link):
        metas = []
        xname = link.x
        yname = link.y
        filemeta = link.get_meta()
        masks = filemeta['masks'].keys()
        if filemeta['columns'] is None:
            metas = [{'name': xname, 'is_multi': False, 'is_nested': False},
                     {'name': yname, 'is_multi': False, 'is_nested': False}]
        else:
            mc = ['dichotomous set', 'categorical set', 'delimited set']
            for name in [xname, yname]:
                if name in filemeta['columns']:
                    dtype = filemeta['columns'][name]['type']
                elif name in filemeta['masks']:
                    dtype = filemeta['masks'][name]['type']
                elif name == '@':
                    dtype = None
                is_multi = True if dtype in mc else False
                is_nested = True if '>' in name else False
                metas.append(
                    {'name': name,
                     'is_multi': is_multi,
                     'is_nested': is_nested,
                     'is_array': name in masks}
                    )
        self._x = metas[0]
        self._y = metas[1]

    def _grp_text_map(self, logic, calc):
        if logic is not None:
            calc_only = self._kwargs.get('calc_only', False)
            net_texts = []
            net_names = []
            for l in logic:
                net_text = l.get('text', None)
                if net_text is not None:
                    del l['text']
                    net_texts.append(net_text)
                else:
                    net_texts.append(None)
                net_names.extend([key for key in l.keys()
                                   if not key == 'expand'])
            grp_text_map = {name: text
                            for name, text in zip(net_names, net_texts)}
            if calc is not None:
                calc_text = calc.get('text', None)
                if calc_text is not None:
                    del calc['text']
                if not calc_only:
                    grp_text_map[calc.keys()[0]] = calc_text
                else:
                    grp_text_map = {calc.keys()[0]: calc_text}
        else:
            grp_text_map = None
        return grp_text_map

    def describe_block(self):
        df = self.dataframe
        logic = self._kwargs['logic']
        global_expand = self._kwargs.get('expand', None)
        block_ref = {}
        if not logic is None:
            for item in logic:
                if isinstance(item, dict):
                    expand = item.get('expand', None)
                    if expand is None:
                        expand = global_expand
                    if expand is None:
                        block_ref[item.keys()[0]] = 'normal'
                    elif expand in ['before', 'after']:
                        for key in item.keys():
                            if not key in ['text', 'expand', 'complete']:
                                net = key
                                break
                        block_ref[net] = 'net'
                        for expanded in item[net]:
                            block_ref[expanded] = 'expanded'
            for idx in df.index.levels[1]:
                if not idx in block_ref:
                    block_ref[idx] = 'normal'

        return block_ref

    def notation(self, method, condition):
        """
        Generate the View's Stack key notation string.

        Parameters
        ----------
        aggname, shortname, relation : str
            Strings for the aggregation name, the method's shortname and the
            relation component of the View notation.

        Returns
        -------
        notation: str
            The View notation.
        """
        notation_strct = 'x|{}|{}|{}|{}|{}'
        axis, _, rel_to, weights, _ = self.get_std_params()
        name = self.name
        if rel_to is None:
            rel_to = ''
        if weights is None:
            weights = ''
        if condition is None:
            condition = ':'
        elif condition in ['x:', ':']:
            condition = condition
        else:
            if not 't.' in method:
                complete = self._kwargs.get('complete', False)
                colon_form = '*:' if complete else ':'
                if axis == 'x':
                    condition = condition + colon_form
                else:
                    condition = colon_form + condition
        return notation_strct.format(method, condition, rel_to, weights, name)

    def get_std_params(self):
        """
        Provides the View's standard kwargs with fallbacks to default values.

        Returns
        -------
        std_parameters : tuple
            A tuple of the common kwargs controlling the general View method
            behaviour: axis, relation, rel_to, weights, text
        """
        return (
            self._kwargs.get('axis', None),
            self._kwargs.get('condition', None),
            self._kwargs.get('rel_to', None),
            self._kwargs.get('weights', None),
            self._kwargs.get('text', '')
            )

    def get_edit_params(self):
        """
        Provides the View's Link edit kwargs with fallbacks to default values.

        Returns
        -------
        edit_params : tuple
            A tuple of kwargs controlling the following supported Link data
            edits: logic, calc, ...
        """
        logic = copy.deepcopy(self._kwargs.get('logic', None))
        calc = copy.deepcopy(self._kwargs.get('calc', None))
        grp_text_map_copy = self.grp_text_map
        if (not logic is None and (isinstance(logic, list) and not
                isinstance(logic[0], dict)) or isinstance(logic, (dict, tuple))):
            logic = [{self.name: logic}]
        self.grp_text_map = self._grp_text_map(logic, calc)
        if not grp_text_map_copy is None:
            self.grp_text_map = grp_text_map_copy
        return (
            logic,
            self._kwargs.get('expand', None),
            self._kwargs.get('complete', False),
            calc,
            self._kwargs.get('exclude', None),
            self._kwargs.get('rescale', None)
            )

    def translate_metric(self, text_key=None, set_value=False):
        if not (self.is_stat() or self.is_base() or self.is_sum()):
            pass
        else:
            text = self.get_std_params()[-1]
            if not self._custom_txt:
                invalid = ['Total', 'Lower quartile', 'Max', 'Min', 'Mean',
                           'Upper quartile', 'Unweighted base', 'Total Sum',
                           'Std. err. of mean', 'Base', 'Median', 'Std. dev',
                           'Sample variance', 'Gross base',
                           'Unweighted gross base', '']
                if not text in invalid:
                    self._custom_txt = text
                    add_custom_text = True
                else:
                    add_custom_text = False
            else:
                add_custom_text = True
            if text_key is None: text_key = 'en-GB'
            transl = self._metric_name_map().get(text_key, 'en-GB')
            try:
                old_val = self.dataframe.index.get_level_values(1)[0]
                custom_txt = self._custom_txt
                if '_gross' in self._notation:
                    if not self.is_weighted():
                        old_val = 'no_w_gross_' + old_val
                    else:
                        old_val = 'gross All'
                elif self.is_base() and not self.is_weighted():
                    old_val = 'no_w_' + old_val
                new_val = transl[old_val]
                if add_custom_text:
                    new_val = new_val + ' ' + custom_txt
                ignore = False
            except (TypeError, KeyError):
                if self.meta()['agg']['text']:
                    new_val = self.meta()['agg']['text']
                else:
                    new_val = old_val
                ignore = True
            if set_value and not ignore:
                if not text == new_val:
                    self._kwargs['text'] = new_val
            else:
                return new_val

    # Currently unused
    # Meant to be used in translate_metric with set_value='index'
    # --> can e.g. replace the inner index value with its translation
    def _update_mi_value(self, axis='x', new_val=None):
        names = ['Question', 'Values']
        q_level = self.dataframe.index.get_level_values(0)[0]
        vals =[q_level, [new_val]]
        self.dataframe.index = pd.MultiIndex.from_product(vals, names=names)
        return None

    def _frequency_condition(self, logic, conditionals, expand):
        axis = self._kwargs.get('axis', 'x')
        if conditionals: conditionals = list(reversed(conditionals))
        logic_codes = []
        for grp in logic:
            if any(isinstance(val, (tuple, dict)) for val in grp.values()):
                codes = conditionals.pop()
                logic_codes.append(codes)
            else:
                expand_cond = expand
                if 'expand' in grp.keys():
                    grp = copy.deepcopy(grp)
                    expand_cond = grp['expand']
                    del grp['expand']
                codes = '{'+','.join(map(str, grp.values()[0]))+'}'
                if expand_cond is None:
                    logic_codes.append("{}[{}]".format(axis, codes))
                elif expand_cond == 'after':
                    logic_codes.append("{}[{}+]".format(axis, codes))
                else:
                    logic_codes.append("{}[+{}]".format(axis, codes))
        return logic_codes

    def _descriptives_condition(self, link):
        if self._kwargs.get('source', None): return self._kwargs['source']
        try:
            if link.x in link.get_meta()['masks'].keys():
                values = link.get_meta()['lib']['values'][link.x]
            else:
                values = link.get_meta()['columns'][link.x].get('values', None)
                if 'lib@values' in values:
                    vals = values.split('@')[-1]
                    values = link.get_meta()['lib']['values'][vals]
            x_values = [int(x['value']) for x in values]
            if self.missing():
                x_values = [x for x in x_values if not x in self.missing()]
            if self.rescaling():
                x_values = [x if not x in self.rescaling()
                            else self.rescaling()[x] for x in x_values]
            if self.missing() or self.rescaling():
                condition = 'x[{}]'.format('{'+','.join(map(str, x_values))+'}')
            else:
                condition = 'x' if self._kwargs.get('axis', 'x') == 'x' else 'y'
        except:
            if self.missing():
                code_excl = '{' + ','.join([str(m) for m in self.missing()]) + '}'
                condition = 'x~{}'.format(code_excl)
            else:
                condition = 'x' if self._kwargs.get('axis', 'x') == 'x' else 'y'
        return condition

    def _calc_condition(self, logic, conditions, calc):
        op = calc.values()[0][1]
        val1, val2 = calc.values()[0][0], calc.values()[0][2]
        symbol_map = {add: '+', sub: '-', mul: '*', div: '/'}
        calc_strct = '{}{}{}'
        if logic:
            cond_names = []
            for l in logic:
                cond_names.extend([key for key in l.keys()
                                   if not key in ['expand', 'text']])
            name_cond_pairs = zip(cond_names, conditions)
            cond_map = {name: cond for name, cond in name_cond_pairs}
            v1 = cond_map[val1] if val1 in cond_map.keys() else val1[0]
            v2 = cond_map[val2] if val2 in cond_map.keys() else val2[0]
        else:
            v1 = val1 if isinstance(val1, list) else conditions
            v2 = val2 if isinstance(val2, list) else conditions
        calc_string = calc_strct.format(v1, symbol_map[op], v2)
        calc_string = calc_string.replace('+{', '{').replace('}+', '}')
        calc_string = calc_string.replace('x', '')
        calc_string = calc_string.replace('[', '').replace(']', '')
        calc_string = 'x[{}]'.format(calc_string)
        return calc_string

    def spec_condition(self, link, conditionals=None, expand=None):
        """
        Updates the View notation's condition component based on agg. details.

        Parameters
        ----------
        link : Link

        Returns
        -------
        relation_string : str
            The relation part of the View name notation.
        """
        logic = self.get_edit_params()[0]
        stat = self._kwargs.get('stats', 'mean')
        complete = self.get_std_params()[2]
        calc = self.get_edit_params()[3]
        if logic is not None:
            condition = self._frequency_condition(logic, conditionals, expand)
        elif stat is not None:
            condition = self._descriptives_condition(link)
        else:
            condition = 'x' if self._kwargs.get('axis', 'x') == 'x' else 'y'
        if calc is not None:
                calc_cond = self._calc_condition(logic, condition, calc)
                if not self._kwargs.get('calc_only', False):
                    if logic:
                        condition = '{},{}'.format(','.join(condition), calc_cond)
                    else:
                        condition = '{},{}'.format(condition, calc_cond)
                else:
                    condition = calc_cond
        else:
            if logic: condition = ','.join(condition)
        return condition

    def missing(self):
        """
        Returns any excluded value codes.
        """
        return self._kwargs.get('exclude', None)

    def rescaling(self):
        """
        Returns the rescaling specification of value codes.
        """
        return self._kwargs.get('rescale', None)

    def weights(self):
        """
        Returns the weight variable name used in the aggregation.
        """
        return self._kwargs.get('weights', None)

    def is_weighted(self):
        """
        Tests if the View is performed on weighted data.
        """
        notation = self._notation.split('|')
        if len(notation[4]) > 0:
            return True
        else:
            return False

    def is_pct(self):
        """
        Tests if the View is a percentage representation of a frequency.
        """
        notation = self._notation.split('|')
        if notation[1] in ['f', 'f.c:f']:
            if len(notation[3]) > 0:
                return True
            else:
                return False
        else:
            return False

    def is_base(self):
        """
        Tests if the View is a base size aggregation.
        """
        notation = self._notation.split('|')
        if notation[1] == 'f':
            if len(notation[2]) == 2:
                return True
            else:
                return False
        else:
            return False

    def is_sum(self):
        """
        Tests if the View is a plain sum aggregation.
        """
        notation = self._notation.split('|')
        if 'f.c' in notation[1]:
            if len(notation[2]) == 2:
                return True
            else:
                return False
        else:
            return False


    def is_net(self):
        """
        Tests if the View is a code group/net aggregation.
        """
        notation = self._notation.split('|')
        if notation[1] in ['f', 'f.c:f']:
            if self._has_code_expr():
                return True
            else:
                return False
        else:
            return False

    def is_counts(self):
        """
        Tests if the View is a count representation of a frequency.
        """
        notation = self._notation.split('|')
        if notation[1] in ['f', 'f.c:f']:
            if len(notation[3]) == 0:
                return True
            else:
                return False
        else:
            return False

    def is_stat(self):
        """
        Tests if the View is a sample statistic.
        """
        if self.meta()['agg']['method'] == 'descriptives':
            return True
        else:
            return False

    def _is_test(self):
        notation = self._notation.split('|')
        if 't.' in notation[1]:
            return True
        else:
            return False

    def is_meanstest(self):
        """
        Tests if the View is a statistical test of differences in means.
        """
        if self._is_test():
            teststr = self._notation.split('|')[1].split('.')
            if teststr[1] == 'means':
                return float(teststr[3].split('+')[0])/100
            else:
                return False
        else:
            return False

    def is_propstest(self):
        """
        Tests if the View is a statistical test of differences in proportions.
        """
        if self._is_test():
            teststr = self._notation.split('|')[1].split('.')
            if teststr[1] == 'props':
                return float(teststr[3].split('+')[0])/100
            else:
                return False
        else:
            return False

    def has_calc(self):
        return 'f.c' in self._notation.split('|')[1]

    def _is_block(self):
        notation = self._notation.split('|')
        if notation[1] in ['f', 'f.c:f']:
            conditions = notation[2].split('[')
            multiple_conditions = len(conditions) > 2
            expand = '+{' in notation[2] or '}+' in notation[2]
            complete = '*:' in notation[2]
            if multiple_conditions or expand or complete:
                return True
            else:
                return False
        else:
            False

    def _has_code_expr(self):
        notation = self._notation.split('|')
        if len(notation[2]) > 3:
            return True
        else:
            return False

    def _shortname(self):
        return self.name.split('|')[-1]

    def _method(self):
        method_part = self._notation.split('|')[1]
        if 'd.' in method_part:
            return 'descriptives'
        elif 'f.' in method_part or method_part == 'f':
            return 'frequency'
        elif 't.' in method_part:
            return 'coltests'
        else:
            return method_part

    @staticmethod
    def _metric_name_map():
        mdict = {
            # English
            'en-GB': {
                '@': 'Total',
                'All': 'Base',
                'no_w_All': 'Unweighted base',
                'gross All': 'Gross base',
                'no_w_gross_All': 'Unweighted gross base',
                'mean': 'Mean',
                'min': 'Min',
                'max': 'Max',
                'median': 'Median',
                'var': 'Sample variance',
                'stddev': 'Std. dev',
                'sem': 'Std. err. of mean',
                'sum': 'Total Sum',
                'lower_q': 'Lower quartile',
                'upper_q': 'Upper quartile'},
            # Danish
            'da-DK': {
                '@': 'Total',
                'All': 'Base',
                'no_w_All': 'Unweighted base',
                'gross All': 'Gross base',
                'no_w_gross_All': 'Unweighted gross base',
                'mean': 'Gennemsnit',
                'min': 'Min',
                'max': 'Max',
                'median': 'Median',
                'var': 'Sample variance',
                'stddev': 'Std.afv',
                'sem': 'StdErr',
                'sum': 'Totalsum',
                'lower_q': 'Nedre kvartil',
                'upper_q': 'Øvre kvartil'},
            # Swedish
            'sv-SE': {
                '@': 'Total',
                'All': 'Bas',
                'no_w_All': 'ovägd bas',
                'gross All': 'Gross base',
                'no_w_gross_All': 'Unweighted gross base',
                'mean': 'Medelvärde',
                'min': 'Min',
                'max': 'Max',
                'median': 'Median',
                'var': 'Sample variance',
                'stddev': 'Std. av.',
                'sem': 'StdErr',
                'sum': 'Summa',
                'lower_q': 'Undre kvartilen',
                'upper_q': 'Övre kvartilen'},
            # Norwegian
            'nb-NO': {
                '@': 'Total',
                'All': 'Base',
                'no_w_All': 'Unweighted base',
                'gross All': 'Gross base',
                'no_w_gross_All': 'Unweighted gross base',
                'mean': 'Gjennomsnitt',
                'min': 'Min',
                'max': 'Max',
                'median': 'Median',
                'var': 'Sample variance',
                'stddev': 'Standardavvik',
                'sem': 'StdErr',
                'sum': 'Totalsum',
                'lower_q': 'Nedre kvartil',
                'upper_q': 'Øvre kvartil'},
            # Finnish
            'fi-FI': {
                '@': 'Total',
                'All': 'Base',
                'no_w_All': 'Unweighted base',
                'gross All': 'Gross base',
                'no_w_gross_All': 'Unweighted gross base',
                'mean': 'Mean',
                'min': 'Min',
                'max': 'Max',
                'median': 'Median',
                'var': 'Sample variance',
                'stddev': 'Std.dev.',
                'sem': 'StdErr',
                'sum': 'Totalsum',
                'lower_q': 'Alakvartiili',
                'upper_q': 'Yläkvartiili'},
            # French
            'fr-FR': {
                '@': 'Total',
                'All': 'Base',
                'no_w_All': 'Base brute',
                'gross All': 'Gross base',
                'no_w_gross_All': 'Unweighted gross base',
                'mean': 'Moyenne',
                'min': 'Min',
                'max': 'Max',
                'median': 'Médiane',
                'var': 'Sample variance',
                'stddev': 'Ecart-type',
                'sem': 'StdErr',
                'sum': 'Totalsum',
                'lower_q': 'Quartile inférieur',
                'upper_q': 'Quartile supérieur'},
            # German
            'de-DE': {
                '@': 'Gesamt',
                'All': 'Basis Netto',
                'no_w_All': 'Ungewichtete Basis Netto',
                'gross All': 'Basis Brutto',
                'no_w_gross_All': 'Ungewichtete Basis Brutto',
                'mean': 'Mittelwert',
                'min': 'Min',
                'max': 'Max',
                'median': 'Median',
                'var': 'Stichprobenvarianz',
                'stddev': 'StdDev',
                'sem': 'StdErr',
                'sum': 'Summe',
                'lower_q': '25% Perzentil',
                'upper_q': '75% Perzentil'}
        }
        for lang in mdict:
            for key in mdict[lang]:
                mdict[lang][key] = mdict[lang][key].decode('utf-8')

        return mdict

    def __repr__(self):
        """ Message to be printed in stdout (print self)

            Example: << View.View Rows: 4, Columns: 3, Has Meta:False >>
        """
        row_count = len(self.dataframe.index)
        columns_count = len(self.dataframe.columns)
        return '%s' % (self.dataframe)
