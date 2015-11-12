import quantipy.core.helpers.functions as helpers
import pandas as pd



class View(object):
    def __init__(self, link, kwargs=None):
        #self._view_attributes = ['meta', 'link', 'dataframe', 'rbases', 'cbases', '_kwargs']
        self._kwargs = kwargs.copy()
        self._link_meta(link)
        self.dataframe = pd.DataFrame()
        self.name = None
        self.rbases = None
        self.cbases = None

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
                     'weights': self.std_params()[3],
                     'method': self._method(),
                     'name': self._shortname(),
                     'fullname': self.name,
                     'text': self.std_params()[4],
                     },
                    'x': self._x,
                    'y': self._y,
                    'shape': self.dataframe.shape
                    }
        return viewmeta

    def std_params(self):
        """
        Provides the View's standard kwargs with fallbacks to default values.
        
        Returns
        -------
        std_parameters : tuple
            A tuple of the common kwargs controlling the general View method
            behaviour: pos, relation, rel_to, weights, text
        """
        return (
            self._kwargs.get('pos', 'x'), 
            self._kwargs.get('relation', None),
            self._kwargs.get('rel_to', None), 
            self._kwargs.get('weights', None),
            self._kwargs.get('text', ''))


    def notation(self, aggname, shortname, relation):
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
        pos, _, rel_to, weights, _ = self.std_params()
        if rel_to is None:
            rel_to = ''
        if weights is None:
            weights = ''
        if relation is None:
            relation = ''
        return '%s|%s|%s|%s|%s|%s' %(pos, aggname, relation, rel_to,
                                     weights, shortname)

    def fulltext_for_stat(self, stat):
        """
        Creates the full text (=label) meta for ``descriptives()`` view
        aggregations. The full text consists of the name of the figure and
        the passed suffix from view method's "text" kwarg.

        Parameters
        ----------
        stat : str
            Name of the stat. figure.

        Returns
        -------
        fulltext : str
            The text that is passed into the meta component of the View.
        """
        texts = {
            'mean': 'Mean',
            'sem': 'Std. err. of mean',
            'median': 'Median',
            'stddev': 'Std. dev.',
            'var': 'Sample variance',
            'varcoeff': 'Coefficient of variation',
            'min': 'Min',
            'max': 'Max'
        }
        text = self.std_params()[-1]
        if text == '':
            self._kwargs['text'] = texts[stat]
        else:
            self._kwargs['text'] = '%s %s' % (texts[stat], self._kwargs['text'])


    def spec_relation(self, link=None):
        """
        Updates the View notation's relation component based on agg. details.
        
        Parameters
        ----------
        link : Link

        Returns
        -------
        relation_string : str
            The relation part of the View name notation.
        """
        logic = self._kwargs.get('logic', None)
        if logic is not None:
            if isinstance(logic, list):
                if isinstance(logic[0], dict):
                    return 'x[%s]:y' % (self._multi_net_string(logic))
                else:
                    return 'x[(%s)]:y' % (self._single_net_string(logic))
        else:
            return self._descriptives_relation(link)



    def _descriptives_relation(self, link):
        try:
            values = link.get_meta()['columns'][link.x].get('values', None)
            if 'lib@values' in values:
                vals = values.split('@')[-1]
                values = link.get_meta()['lib']['values'][vals]
                x_values = [int(x['value']) for x in values]
            else:
                x_values = [int(x['value']) for x in
                          link.get_meta()['columns'][link.x]['values']]
            if self.missing():
                x_values = [x for x in x_values if not x in self.missing()]
            if self.rescaling():
                x_values = [x if not x in self.rescaling() else self.rescaling()[x] for x in x_values]
            if self.missing() or self.rescaling():
                relation = 'x%s:y' % (str(x_values).replace(' ', ''))
            else:
                relation = 'x:y'
        except:
            relation = 'x:y'
    
        return relation          
    
    def _multi_net_string(self, logic):
        return (','.join([str(tuple(grp.values()[0])).replace(' ', '')
                         for grp in logic])).replace(',)', ')')

    def _single_net_string(self, logic):
        return ','.join([str(c) for c in logic])


   
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
        notation = self.name.split('|')
        if len(notation[4]) > 0:
            return True
        else:
            return False

    def is_pct(self):
        """
        Tests if the View is a percentage representation of a frequency.
        """
        notation = self.name.split('|')
        if notation[1] == 'frequency':
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
        notation = self.name.split('|')
        if notation[1] == 'frequency':
            if len(notation[2]) == 3:
                return True
            else:
                return False
        else:
            return False


    def is_net(self):
        """
        Tests if the View is a code group/net aggregation.
        """
        notation = self.name.split('|')
        if notation[1] == 'frequency':
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
        notation = self.name.split('|')
        if notation[1] == 'frequency':
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
        notation = self.name.split('|')
        if 'tests' in notation[1]:
            return True
        else:
            return False

    def is_meanstest(self):
        """
        Tests if the View is a statistical test of differences in means.
        """
        if self._is_test():
            teststr = self.name.split('|')[1].split('.')
            if teststr[1] == 'means':
                return float(teststr[3])/100
            else:
                return False
        else:
            return False

    def is_propstest(self):
        """
        Tests if the View is a statistical test of differences in proportions.
        """
        if self._is_test():
            teststr = self.name.split('|')[1].split('.')
            if teststr[1] == 'props':
                return float(teststr[3])/100
            else:
                return False
        else:
            return False


    def _link_meta(self, link):
        metas = []
        xname = link.x 
        yname = link.y
        filemeta = link.get_meta()
        if filemeta['columns'] is None:
            metas = [{'name': xname, 'is_multi': False, 'is_nested': False},
                     {'name': yname, 'is_multi': False, 'is_nested': False}]
        else:
            mc = ['dichotomous set', 'categorical set', 'delimited set']
            
            for name in [xname, yname]:
                if name in filemeta['columns']:
                    type = filemeta['columns'][name]['type']
                elif name in filemeta['masks']:
                    type = filemeta['masks'][name]['type']
                elif name == '@':
                    type = None
                is_multi = True if type in mc else False
                is_nested = True if '>' in name else False
                metas.append(
                    {'name': name, 'is_multi': is_multi, 'is_nested': is_nested}
                    )
        self._x = metas[0]
        self._y = metas[1]

    def _has_code_expr(self):
        notation = self.name.split('|')
        if len(notation[2]) > 3:
            return True
        else:
            return False

    def _shortname(self):
        return self.name.split('|')[-1]

    def _method(self):
        method_part = self.name.split('|')[1]
        if method_part in ['mean', 'median', 'var', 'stddev', 'varcoeff',
                           'sem', 'max', 'min']:
            return 'descriptives'
        elif 'tests' in method_part:
            return 'coltests'
        else:
            return method_part


    def __repr__(self):
        """ Message to be printed in stdout (print self)

            Example: << View.View Rows: 4, Columns: 3, Has Meta:False >>
        """
        row_count = len(self.dataframe.index)
        columns_count = len(self.dataframe.columns)
        return '%s' % (self.dataframe)
