import io
import sys
import numpy as np
import pandas as pd
from rim import Rim
from collections import OrderedDict


class WeightEngine:

    def __init__(self,
                 data=None,
                 dropna=True,
                 meta=None
                 ):

        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "\n You must pass a pandas.DataFrame to the 'data' argument of the WeightEngine"
                "\n constructor. If your DataFrame is serialized please load it first."
                )
        
        self._df = data.copy()

        self.schemes = {}

        self.original_columns = None

        # Detect missing value markers (empty strings and the value of na_values)
        # In data without any NAs, passing dropna=False can improve the performance of reading a large file
        self.dropna = dropna

        # Constants
        self._SCHEME = 'scheme'
        self._KEY = 'key'

        if meta is not None:
            if not isinstance(meta, (dict, list)):
                raise ValueError(
                    "\n You must pass a dict or list to the 'meta' argument of the WeightEngine"
                    "\n constructor. If your meta is serialized please load it first."
                    )
            self._meta = meta

    def get_report(self):
        """
        Return a DataFrame summarising results of the calculated weights.

        Used to return the metrics commonly used to measure the
        effectiveness of the calculated weights. Metrics returned
        (below) will be broken down by weight variable and weight
        filter.

        - Total: unweighted
        - Total: weighted
        - Weighting efficiency
        - Iterations required
        - Mean weight factor
        - Minimum weight factor
        - Maximum weight factor
        - Weight factor ratio

        Parameters
        ----------
        self : self

        Returns
        -------
        None
        """

        # List to store each report series for later concatenation
        reports = []

        for scheme_name in sorted(self.schemes.keys()):
            scheme = self.schemes[scheme_name]['scheme']
            data = self.dataframe(scheme_name)

            for group_name in sorted(scheme.groups.keys()):
                report = pd.Series([])
                weight_col = 'weights_{0}'.format(scheme_name)
                group = scheme.groups[group_name]
                filter_def = group['filters']

                report['Weight variable'] = weight_col
                report['Weight group'] = group_name

                if filter_def is None:
                    report['Weight filter'] = 'None'
                    # This is a trick to keep rows that are not NaN
                    # since Nan is never equal to NaN.
                    filter_def = '{0}=={0}'.format(weight_col)
                else:
                    report['Weight filter'] = filter_def

                filtered_data = data.query(filter_def)
                weight_factors = filtered_data[weight_col]

                weight_count = weight_factors.count()
                weight_sum = weight_factors.sum()

                # The weighting efficiency is normally calculated by
                # first dividing the squared sum of the weights by the
                # number of respondents, then dividing that by the sum
                # of the squared weights, however, in order to account
                # for controlling the total weighted base the
                # calculation here is changed to multiplying the sum of
                # the weights divided by the number of respondents by
                # the sum of the weights divided by the sum of the
                # squared weights. If you don't make this change the
                # weighting efficiency for schemes with controlled
                # bases will be incorrect.
                efficiency = (
                    (weight_sum / weight_count) * (
                        weight_sum / weight_factors.pow(2).sum()
                    )
                ) * 100
                mean = filtered_data[weight_col].mean()
                minimum = filtered_data[weight_col].min()
                maximum = filtered_data[weight_col].max()

                report['Total: unweighted'] = weight_count
                report['Total: weighted'] = weight_sum
                report['Weighting efficiency'] = efficiency
                report['Iterations required'] = group['iterations']
                report['Mean weight factor'] = mean
                report['Minimum weight factor'] = minimum
                report['Maximum weight factor'] = maximum
                report['Weight factor ratio'] = maximum / minimum

                reports.append(report)

        report_df = pd.DataFrame(reports).set_index([
            'Weight variable',
            'Weight group',
            'Weight filter'
        ]).T
        return report_df

    def run(self, schemes=[]):
        if isinstance(schemes, (str, unicode)):
            schemes = [schemes]
        if isinstance(schemes, list):

            if len(schemes) == 0:  # Weight all schemes
                schemes = self.schemes

            for scheme in schemes:
                if scheme in self.schemes:
                    the_scheme = self.schemes[scheme][self._SCHEME]

                    weights = the_scheme._compute()
                    self._df[the_scheme._weight_name()] = weights
    
                else:
                    raise Exception(("Scheme '%s' not found." % scheme))
        else:
            raise ValueError(('schemes must be of type %s NOT %s ') % (type([]), type(scheme)))

    def report(self, scheme, group=None):
        report = self.schemes[scheme][self._SCHEME].report(group)
        group_names = sorted(report.keys())
        summary_df = pd.DataFrame([report[gn]['summary'] for gn in group_names]).T
        idx_tuples = zip(*[summary_df.columns, group_names])
        summary_df.columns = pd.MultiIndex.from_tuples(idx_tuples, names=['Weight variable', 'Weight group'])
        report['summary'] = summary_df
        return report

    def dataframe(self, scheme=None):
        if scheme is None:
            # Return the whole dataframe if no scheme is selected
            return self._df
        elif isinstance(scheme, (str, unicode)):
            if scheme in self.schemes:
                the_scheme = self.schemes[scheme][self._SCHEME]
                key_column = self.schemes[scheme][self._KEY]
                return the_scheme.dataframe(self._df, key_column=key_column)
            else:
                raise Exception("Scheme not found.")
        else:
            raise ValueError(
                (
                    'scheme must be of type %s, %s or %s NOT %s '
                ) % (type(str), type(unicode), type(None), type(scheme))
            )

    def add_scheme(self, scheme, key, verbose=True):
        if scheme.name in self.schemes:
            print "Overwriting existing scheme '%s'." % scheme.name
        self.schemes[scheme.name] = {self._SCHEME: scheme, self._KEY: key}
        scheme._minimize_columns(self._df, key, verbose)
