import io
import sys
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
            self._verify_metadata()

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
                    self._df[the_scheme.weight_name()] = weights
    
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
        elif isinstance(scheme, str):
            if scheme in self.schemes:
                the_scheme = self.schemes[scheme][self._SCHEME]
                key_column = self.schemes[scheme][self._KEY]
                return the_scheme.dataframe(self._df, key_column=key_column)
            else:
                raise Exception("Scheme not found.")
        else:
            raise ValueError(('scheme must be of type %s or %s NOT %s ') % (type(str), type(None), type(scheme)))

    def add_scheme(self, scheme, key):
        if scheme.name in self.schemes:
            print "Overwriting existing scheme '%s'." % scheme.name
        self.schemes[scheme.name] = {self._SCHEME: scheme, self._KEY: key}
        scheme.minimize_columns(self._df, key)