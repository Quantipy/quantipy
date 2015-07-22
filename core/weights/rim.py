
import pandas as pd
import numpy as np
import io
import re
import itertools
import pdb
import copy

class Rim:
    def __init__(self,
                 name,
                 lists=[],
                 max_iterations=1000,
                 convcrit=0.01,
                 cap=0,
                 dropna=True,
                 impute_method="mean",
                 weight_column_name=None,
                 total=0
                 ):

        # Default var init
        self.name = name
        self.type = "Rim"
        self.target_cols = lists
        self.max_iterations = 1000
        self.convcrit = 0.01
        self.cap = cap
        self.weight_column_name = weight_column_name
        self.total = total

        self._group_targets = {}

        # Store a (sub)dataframe in the scheme
        self._df = None

        # Impute methods parameters
        self.dropna = dropna
        self._impute_method = impute_method
        self._impute_method_specific = {}  # Where key is column name and value is method name (e. {'q_06':'mean'})

        # Constants
        self._FILTER_DEF = 'filters'
        self._TARGETS = 'targets'
        self._TARGETS_INDEX = 'targets_index'
        self._REPORT = 'report'
        self._DEFAULT_GROUP_NAME = '_default_group_name_'
        self._WEIGHTS_ = 'weights_'

        # default group init
        # a group can have any name except for the _DEFAULT_GROUP_NAME
        # _DEFAULT_GROUP_NAME is used when no group is defined for the weight scheme
        self.groups = {}
        self.groups[self._DEFAULT_GROUP_NAME] = {}
        self.groups[self._DEFAULT_GROUP_NAME][self._REPORT] = None
        self.groups[self._DEFAULT_GROUP_NAME][self._FILTER_DEF] = None
        self.groups[self._DEFAULT_GROUP_NAME][self._TARGETS] = self._empty_target_list()
        self.groups[self._DEFAULT_GROUP_NAME][self._TARGETS_INDEX] = None

    def set_targets(self, targets, group_name=None):
        """
        Quickly set simple weight targets. optionally assigning a group name.

        Paramters
        ---------
        targets : dict
            Dictionary mapping of DataFrame columns to target proportion list.
        group_name : str, optional
            A name for the simple weight (group) created.

        Returns
        -------
        None
        """
        gn = self._DEFAULT_GROUP_NAME if group_name is None else group_name 
        if group_name is not None and self._DEFAULT_GROUP_NAME in self.groups.keys():
            self.groups[gn] = self.groups.pop(self._DEFAULT_GROUP_NAME)
        if not isinstance(targets, dict):
            raise ValueError(("'targets' must be of type dict, '%s' was given.")\
                             % type(targets))
        else:
            self.groups[gn][self._TARGETS] = {}
            for target_col, target_props in targets.iteritems():
                if not isinstance(target_props, list):
                    raise ValueError("'%s's target proportions must be of ' \
                        'type list, %s' was given." % (target_col,
                                                       type(target_props)))
                else:
                    if not target_col in self.target_cols:
                        self.target_cols.append(target_col)
                    self.groups[gn][self._TARGETS][target_col] = target_props

    def add_group(self, name=None, filter_def=None, targets=None):
        """
        DOCSTRING

        Parameters
        ----------

        Returns
        -------
        """
        if name is None:
            gn = self._DEFAULT_GROUP_NAME
            use_default_name = True
        else:
            gn = name
            use_default_name = False
        if gn not in self.groups:
            self.groups[gn] = {}
        if self._TARGETS not in self.groups[gn]:
            self.groups[gn][self._TARGETS] = self._empty_target_list()
        if use_default_name and len(self.groups.keys()) > 1:
            del self.groups[self._DEFAULT_GROUP_NAME]
        if targets is not None:
            self.set_targets(targets=targets, group_name=gn)
        self.groups[gn][self._FILTER_DEF] = filter_def

    def _compute(self):
        self._get_base_factors()
        if self._group_targets.keys():
            self._adjust_groups()
        if self.total > 0 and not self._group_targets.keys():
            self._scale_total()
        return self._df[self.weight_name()]

    def _get_base_factors(self):
        for group in self.groups:
            wdf = self._get_wdf(group)
            wdf[self.weight_name()] = 1
            rake = Rake(wdf, self.groups[group][self._TARGETS],
                        self.weight_name(), use_cap=self.use_cap(),
                        cap=self.cap)
            rake.start()
            self.groups[group][self._REPORT] = rake.report
            self._df.loc[rake.dataframe.index, self.weight_name()] = \
            rake.dataframe[self.weight_name()]
        return self._df[self.weight_name()]

    def _scale_total(self): 
        weight_var = self.weight_name()
        self._df[weight_var].replace(1, 0, inplace=True)
        self._df[weight_var] = (self._df[weight_var] /
                                self._df[weight_var].sum() * self.total)
        for group in self.groups:
            filter_def = self.groups[group][self._FILTER_DEF]
            if filter_def is not None:
                self.groups[group]['report']['summary']['Total: weighted'] = \
                self._df.query(filter_def)[weight_var].sum()
            else:
                self.groups[group]['report']['summary']['Total: weighted'] = \
                self._df[weight_var].sum()

    def _adjust_groups(self):
        adj_w_vec = pd.Series()
        if self.total == 0:
            self.total = sum([len(self._get_wdf(group).index)
                              for group in self.groups])
        for group in self.groups:
            w_vec = self._get_wdf(group)[self.weight_name()]
            sub_weight_sum = w_vec.sum()
            ratio = self._group_targets[group] / (sub_weight_sum / self.total)
            self.groups[group]['report']['summary']['Total: weighted'] = \
            (w_vec * ratio).sum()

            adj_w_vec = adj_w_vec.append(w_vec * ratio).dropna()
        self._df[self.weight_name()] = adj_w_vec

    def _get_filter_cols(self, filter_def):
        return [colname for colname in self._df.columns
                if colname in filter_def]

    def _get_target_cols(self, targets):
        return targets.keys()

    def _get_wdf(self, group):
        filters = self.groups[group][self._FILTER_DEF]
        targets = self.groups[group][self._TARGETS]
        target_vars = self._get_target_cols(targets)
        weight_var = self.weight_name()
        if filters is not None:
            wdf = self._df.copy().query(filters)
            filter_vars = self._get_filter_cols(filters)
            selected_cols = target_vars + filter_vars + [weight_var]
        else:
            wdf = self._df.copy()
            selected_cols = target_vars + [weight_var]
        wdf = wdf[selected_cols].dropna()
        return wdf

    def report(self, group=None):
        report = {}
        if group is None:
            for group in self.groups:
                report[group] = self.groups[group][self._REPORT]
        else:
            report[group] = self.groups[group][self._REPORT]
        return report

    def impute_method(self, method, target=None):
        if target is None:
            self._impute_method = method
        else:
            self._impute_method_specific[target] = method

    def minimize_columns(self, df, key):
        group_filter_columns = []
        for group in self.groups:
            if isinstance(self.groups[group]['filters'], dict):
                filter = self.groups[group]['filters'].keys()[0]
                if group_filter_columns.count(filter) == 0:
                    group_filter_columns.append(filter)
        columns = [key] + self.target_cols + group_filter_columns
        self._df = pd.DataFrame(df, copy=True)
        self._df[self.weight_name()] = pd.np.zeros(len(self._df))
        self._check_targets()

    def dataframe(self, df, key_column=None):
        all_filter_cols = [self._get_filter_cols(
                            self.groups[group][self._FILTER_DEF])
                           for group in self.groups]
        all_filter_cols = list(set([filter_col for sublist in all_filter_cols
                                   for filter_col in sublist]))
        columns = self.columns(add_columns=[key_column])
        columns.extend(all_filter_cols)
        df = df.copy()[columns]
        df[self.weight_name()].replace(0, np.NaN, inplace=True)
        df.dropna(subset=[self.weight_name()], inplace=True)
        return df
            
    def columns(self, identifier=None, add_columns=None):
        if identifier is not None:
            columns = [identifier]
        else:
            columns = []
        if add_columns:
            columns += add_columns
        [columns.append(target_col) for target_col in self.target_cols]
        columns.append(self.weight_name())
        return columns

    def use_cap(self):
        if self.cap <= 0:
            return False
        else:
            return True

    def group_targets(self, group_targets):
        if isinstance(group_targets, dict):
            for group in group_targets:
                if group_targets[group] < 1:
                    self._group_targets[group] = group_targets[group]
                else:
                    self._group_targets[group] = group_targets[group] / 100.0
        else:
            raise ValueError(('Group_targets must be of type %s NOT %s ') % \
                             (type({}), type(group_targets)))

    def weight_name(self):
        if self.weight_column_name is None:
            return self._WEIGHTS_ + self.name
        else:
            return self.weight_column_name

    def _empty_target_list(self):
        return {list_item: [] for list_item in self.target_cols}

    def _dropna(self):
        if self.dropna:
            self._df.dropna(how='any', inplace=True)
        else:
            columns = self._impute_method_specific
            columns.update({column: self._impute_method for column in self.target_cols if column not in self._impute_method_specific.keys()})

            for column, method in columns.iteritems():
                if method == "mean":
                    self._df[column].fillna(self._df[column].mean(), inplace=True)
                elif method == "mode":
                    self._df[column].fillna(self._df[column].mode()[0], inplace=True)

    def _check_targets(self):
        """
        Check correct weight variable input proportion lengths and sum of 100.
        """
        len_err = 'Scheme "{0}", group "{1}": The targets for the variable '\
                  '"{2}" do not match the number of unique value codes. It '\
                  'should have been {3}, got {4}.'
        sum_err = 'Scheme "{0}", group "{1}": The targets for the variable '\
                  '"{2}" do not add up to 100. Sum is: {3}'
        for group in self.groups:
            clean_df = self._df[self.groups[group][self._TARGETS].keys()].dropna()
            for target_col, target_props in self.groups[group][self._TARGETS].items():
                unique_codes = pd.unique(clean_df[target_col]).tolist()
                if not len(target_props) == len(unique_codes):
                    raise ValueError(len_err.format(self.name, group,
                                     target_col, len(unique_codes),
                                     len(target_props)))
                if not np.allclose(np.sum(target_props), 100.0):
                    raise ValueError(sum_err.format(self.name, group,
                                     target_col, np.sum(target_props)))

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

class Rake:
    def __init__(self, dataframe, targets,
                 weight_column_name="weight",
                 max_iterations=1000,
                 convcrit=0.01,
                 use_cap=False,
                 cap=10000000,
                 verbose=False):

        self.targets = targets
        self.dataframe = dataframe
        self.weight_column_name = weight_column_name

        self.cap = cap
        self.use_cap = use_cap
        self.max_iterations = max_iterations
        self.convcrit = convcrit

        self.report = {}
        self.iteration_counter = 0  # for the report

        #do we print out extra information
        self.verbose = verbose

        #Parse the dataframe
        if isinstance(dataframe, pd.DataFrame):
            self.dataframe = dataframe
        else:
            raise Exception("Unknown data type (%s). Should be <pandas.DataFrame>.", type(dataframe))

        #this creates the weight vector [a series with ones]
#        self.dataframe[self.weight_column_name] = pd.np.ones(len(self.dataframe))
        self.pre_weight = pd.np.ones(len(self.dataframe))

        #Parse the targets
        self.rowcount = len(self.dataframe)
        if isinstance(targets, dict):
            if isinstance(targets[targets.keys()[0]][0], int) or isinstance(targets[targets.keys()[0]][0], float):
                targets = {key: [float(i)/100*self.rowcount for i in targets[key]] for key in targets.keys()}
            else:
                targets = {key: [float(i)/100*self.rowcount for i in targets[key].split(';')] for key in targets.keys()}
            self.targets = targets

        self.keys = targets.keys()
        self.keys_row = self.keys[0:len(self.keys):2]
        self.keys_col = self.keys[1:len(self.keys):2]

        if pd.np.isnan(self.dataframe[self.weight_column_name]).sum() > 0:
            raise Exception("seed weights cannot have missing values, use filter to eliminate missing values or substitute 1 for missing cases")
        if cap <= 1 and use_cap:
            raise Exception("cap may not be less than or equal to 1")
        if cap < 1.5 and use_cap:
            print "cap is very low, the model may take a long time to run"

    def rakeonvar(self, target):
        #start=1 : This assumes that survey data ALWAYS starts with 1 (eg, sex: male=1, female=2, undisclosed=3)
        #index is the enumerator (1, 2, 3) and weight is the weight value [50.0, 50.0, 0.0]
        for index, weight in enumerate(self.targets[target], start=1):
            df = self.dataframe[self.dataframe[target] == index]  # df is a subset of the dataframe for the current target
            index_array = (self.dataframe[target] == index)
            data = df[self.weight_column_name] * (weight / sum(df[self.weight_column_name]))
#            pdb.set_trace()
            self.dataframe.loc[index_array, self.weight_column_name] = data

    def calc_weight_efficiency(self):
        numerator = 100*sum(self.dataframe[self.weight_column_name] *
                            self.pre_weight) ** 2
        denominator = (sum(self.pre_weight) *
                       sum(self.pre_weight*
                           self.dataframe[self.weight_column_name] ** 2))
        self.weight_efficiency = numerator / denominator
        return self.weight_efficiency

    def generate_report(self):
        weights = self.dataframe[self.weight_column_name]
        r_summary = [
            {"Total: unweighted": weights.count()},
            {"Total: weighted": weights.sum()},
            {"Weighting efficiency": self.calc_weight_efficiency()},
            {"Iterations required": self.iteration_counter},
            {"Minimum weight factor": weights.min()},
            {"Maximum weight factor": weights.max()},
            {"Weight factor ratio": weights.max() / weights.min()}
        ]

        self.report['summary'] = pd.Series(pd.concat([pd.Series(s) for s in r_summary]), name=self.weight_column_name)

        self.report["targets"] = self.targets

        # The data is a representation/manipulation of the dataframe
        self.report["data"] = {}
        self.report["data"]["factor weights"] = self.dataframe.pivot_table(index=self.keys_row, columns=self.keys_col, values=self.weight_column_name, dropna=False, fill_value=0)
        self.report["data"]["input"] = {}
        self.report["data"]["input"]["absolute"] = self.dataframe[self.keys].pivot_table(index=self.keys_row, columns=self.keys_col, aggfunc=len, dropna=False, fill_value=0)
        self.report["data"]["input"]["relative"] = self.report["data"]["input"]["absolute"] / self.rowcount
        self.report["data"]["output"] = {}
        self.report["data"]["output"]["absolute"] = self.report["data"]["input"]["absolute"] * self.report["data"]["factor weights"]
        self.report["data"]["output"]["relative"] = self.report["data"]["output"]["absolute"] / self.rowcount

    def start(self):
        pct_still = 1 - self.convcrit
        diff_error = 999999
        diff_error_old = 99999999999
        for iteration in range(1, self.max_iterations+1):
            old_weights = self.dataframe[self.weight_column_name].copy()

            if not diff_error < pct_still * diff_error_old:
                break

            for target in self.targets:
                self.rakeonvar(target)

            if self.use_cap:
                #cap (this needs more rigorous testings)
                while self.dataframe[self.weight_column_name].max() > self.cap + 0.0001:
                    self.dataframe.loc[self.dataframe[self.weight_column_name] > self.cap, self.weight_column_name] = self.cap
                    self.dataframe[self.weight_column_name] = self.dataframe[self.weight_column_name]/pd.np.mean(self.dataframe[self.weight_column_name])

            diff_error_old = diff_error
            diff_error = sum(abs(self.dataframe[self.weight_column_name]-old_weights))

        self.iteration_counter = iteration  # for the report

        if iteration == self.max_iterations:
            print 'Convergence did not occur in %s iterations' % iteration
        else:
            if diff_error > 0.001:
                print "Raking achieved only partial convergence, please check the results to ensure that sufficient convergence was achieved."
                print "No improvement was apparent after %s iterations" % iteration
            else:
                if self.verbose:
                    print 'Raking converged in %s iterations' % iteration
                    print 'Generating report'
                self.generate_report()
