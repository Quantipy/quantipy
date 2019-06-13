
import pandas as pd
import numpy as np
import io
import re
import itertools
import pdb
import copy
import warnings
import time

from quantipy.core.tools.view.logic import (
    has_any, has_all, has_count,
    not_any, not_all, not_count,
    is_lt, is_ne, is_gt,
    is_le, is_eq, is_ge,
    union, intersection, get_logic_index)


class Rim:
    def __init__(self,
                 name,
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
        self.target_cols = []
        self.max_iterations = max_iterations
        self.convcrit = 0.01
        self.cap = cap
        self.weight_column_name = weight_column_name
        self.total = total
        self.anesrake_cap_correction = True

        self._group_targets = {}

        # Storage of weight (sub-)dataframe
        self._df = None

        # Impute methods parameters
        self.dropna = dropna
        self._impute_method = impute_method
        self._specific_impute = {}

        # Constants
        self._FILTER_DEF = 'filters'
        self._FILTER_DEF_ORG = 'filters_org'
        self._FILTER_VARS = 'filter_vars'
        self._TARGETS = 'targets'
        self._TARGETS_INDEX = 'targets_index'
        self._REPORT = 'report'
        self._DEFAULT_NAME = '_default_name_'
        self._WEIGHTS_ = 'weights_'
        self._ITERATIONS_ = 'iterations'

        # Default group init
        # A group can have any name except for the _DEFAULT_NAME
        # "_DEFAULT_NAME" is used when no group name is provided
        self.groups = {}
        self.groups[self._DEFAULT_NAME] = {}
        self.groups[self._DEFAULT_NAME][self._REPORT] = None
        self.groups[self._DEFAULT_NAME][self._FILTER_DEF] = None
        self.groups[self._DEFAULT_NAME][self._FILTER_DEF_ORG] = None
        self.groups[self._DEFAULT_NAME][self._FILTER_VARS] = []
        self.groups[self._DEFAULT_NAME][self._TARGETS] = self._empty_target_list()
        self.groups[self._DEFAULT_NAME][self._TARGETS_INDEX] = None
        self.groups[self._DEFAULT_NAME][self._ITERATIONS_] = None

    def set_targets(self, targets, group_name=None):
        """
        Quickly set simple weight targets, optionally assigning a group name.

        Parameters
        ----------
        targets : dict or list of dict
            Dictionary mapping of DataFrame columns to target proportion list.
        group_name : str, optional
            A name for the simple weight (group) created.

        Returns
        -------
        None
        """
        if not isinstance(targets, list): targets = [targets]
        gn = self._DEFAULT_NAME if group_name is None else group_name
        if group_name is not None and self._DEFAULT_NAME in list(self.groups.keys()):
            self.groups[gn] = self.groups.pop(self._DEFAULT_NAME)

        mul_targets_err = 'Multiple weight targets must be given as list of dicts,\n'\
                          'input is {}'.format(type(targets))

        target_map_err = 'Weight targets must be given as dicts of dict:\n' \
                         'mapping {column name: {code: proportion}}.'
        if isinstance(targets, dict) and len(list(targets.keys())) > 1:
            raise TypeError(mul_targets_err)

        for target in targets:
            if not isinstance(target, dict) or not isinstance(list(target.values())[0], dict):
                raise TypeError(target_map_err)
        self.groups[gn][self._TARGETS]  = {}
        for target in targets:
            if not list(target.keys())[0] in self.target_cols:
                self.target_cols.append(list(target.keys())[0])
            self.groups[gn][self._TARGETS] = targets


    def add_group(self, name=None, filter_def=None, targets=None):
        """
        Set weight groups using flexible filter and target defintions.

        Main method to structure and specify complex weight schemes.

        Parameters
        ----------
        name : str
            Name of the weight group.
        filter_def : str, optional
            An optional filter defintion given as a boolean expression in
            string format. Must be a valid input for the pandas
            DataFrame.query() method.
        targets : dict
            Dictionary mapping of DataFrame columns to target proportion list.

        Returns
        -------
        None
        """
        if name is None:
            gn = self._DEFAULT_NAME
            use_default_name = True
        else:
            gn = name
            use_default_name = False
        if gn not in self.groups:
            self.groups[gn] = {}
        if self._TARGETS not in self.groups[gn]:
            self.groups[gn][self._TARGETS] = self._empty_target_list()
        if use_default_name and len(list(self.groups.keys())) > 1:
            del self.groups[self._DEFAULT_NAME]
        if targets is not None:
            self.set_targets(targets=targets, group_name=gn)
        self.groups[gn][self._FILTER_DEF] = filter_def
        self.groups[gn][self._FILTER_DEF_ORG] = filter_def

    def _compute(self):
        self._get_base_factors()
        self._df[self._weight_name()].replace(0.00, 1.00, inplace=True)
        self._df[self._weight_name()].replace(-1.00, np.NaN, inplace=True)
        if list(self._group_targets.keys()):
            self._adjust_groups()
        if self.total > 0 and not list(self._group_targets.keys()):
            self._scale_total()
        for group in self.groups:
            filter_def = self.groups[group][self._FILTER_DEF]
            try:
                if filter_def is not None:
                    self.groups[group]['report']['summary']['Total: weighted'] = \
                    self._df.query(filter_def)[self._weight_name()].sum()
                    self.groups[group]['report']['summary']['Total: unweighted'] = \
                    self._df.query(filter_def)[self._weight_name()].count()
                else:
                    self.groups[group]['report']['summary']['Total: weighted'] = \
                    self._df[self._weight_name()].sum()
                    self.groups[group]['report']['summary']['Total: unweighted'] = \
                    self._df[self._weight_name()].count()
            except Exception as e:
                warn = 'Could not properly adjust Totals in report!'
                warnings.warn(warn)
        return self._df[self._weight_name()]

    def _get_base_factors(self):
        wgt = self._weight_name()
        for group in self.groups:
            wdf = self._get_wdf(group)
            wdf[wgt] = 1
            rake = Rake(wdf,
                        self.groups[group][self._TARGETS],
                        self._weight_name(),
                        max_iterations = self.max_iterations,
                        _use_cap=self._use_cap(),
                        cap=self.cap,
                        anesrake_cap_correction=self.anesrake_cap_correction)
            self.groups[group][self._ITERATIONS_] = rake.start()
            self._df.loc[rake.dataframe.index, wgt] = rake.dataframe[wgt]
            self.groups[group][self._REPORT] = rake.report
        invalid_idx = []
        for group in self.groups:
            if self.groups[group][self._FILTER_DEF] is not None:
                invalid_idx.extend(
                    self._df.query(self.groups[group][self._FILTER_DEF]).index)
        if invalid_idx:
            filter_idx = [idx for idx in self._df.index
                          if idx not in invalid_idx]
        else:
            filter_idx = invalid_idx
        self._df.loc[filter_idx, wgt] = -1.00
        return None

    def _scale_total(self):
        weight_var = self._weight_name()
        self._df[weight_var].replace(1.00, np.NaN, inplace=True)
        unw_total = len(self._df[weight_var].dropna().index)
        self._df[weight_var].replace(np.NaN, 0.00, inplace=True)
        scale_factor = float(unw_total) / float(self.total)
        self._df[weight_var] = self._df[weight_var] / scale_factor
        self._df[weight_var].replace(0.00, 1.00, inplace=True)


    def _adjust_groups(self):
        adj_w_vec = pd.Series()
        for group in self.groups:
            w_vec = self._df.query(self.groups[group][self._FILTER_DEF])[self._weight_name()]
            if self.total > 0:
                ratio = float(self._group_targets[group]) * w_vec
                scale_factor = len(w_vec.index) / float(self.total)
                ratio = ratio / scale_factor
                #self.groups[group]['report']['summary']['Total: weighted'] = ratio.sum()
            else:
                valid_counts = self._df[self._weight_name()].count()
                ratio = float(self._group_targets[group]) * w_vec
                scale_factor = len(w_vec.index) / float(valid_counts)
                ratio = ratio / scale_factor
                #self.groups[group]['report']['summary']['Total: weighted'] = ratio.sum()
            adj_w_vec = adj_w_vec.append(ratio).dropna()
        self._df[self._weight_name()] = adj_w_vec


    def _get_group_filter_cols(self, filter_def):
        filter_cols = []
        if filter_def is not None:
            for colname in self._df.columns:
                if re.search(r"\b"+colname+r"\b", filter_def):
                    filter_cols.append(colname)
        return filter_cols

    def _get_group_target_cols(self, targets):
        return [list(target.keys())[0] for target in targets]

    def _get_wdf(self, group):
        filters = self.groups[group][self._FILTER_DEF]
        targets = self.groups[group][self._TARGETS]
        target_vars = self._get_group_target_cols(targets)
        weight_var = self._weight_name()
        if filters is not None:
            wdf = self._df.copy().query(filters)
            filter_vars = self.groups[group][self._FILTER_VARS]
            selected_cols = target_vars + filter_vars + [weight_var]
        else:
            wdf = self._df.copy()
            selected_cols = target_vars + [weight_var]
        wdf = wdf[selected_cols].dropna()
        return wdf

    def _dropna(self):
        if self.dropna:
            self._df.dropna(inplace=True)
        else:
            columns = self._specific_impute
            columns.update({column: self._impute_method
                            for column in self.target_cols
                            if column not in list(self._specific_impute.keys())})

            for column, method in columns.items():
                if method == "mean":
                    m = np.round(self._df[column].mean(), 0)
                    print(m)
                    self._df[column].fillna(m, inplace=True)
                elif method == "mode":
                    self._df[column].fillna(self._df[column].mode()[0], inplace=True)

    def report(self, group=None):
        """
        TODO: Docstring
        """
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
            self._specific_impute[target] = method

    def _get_scheme_filter_cols(self):
        scheme_filter_cols = [self.groups[group][self._FILTER_VARS]
                              for group in self.groups]
        scheme_filter_cols = list(set([filter_col
                                       for sublist in scheme_filter_cols
                                       for filter_col in sublist]))
        return scheme_filter_cols

    def _minimize_columns(self, df, key, verbose=True):
        self._df = df.copy()
        filter_cols = self._get_scheme_filter_cols()
        columns = list(set([key] + self.target_cols + filter_cols))
        self._df = self._df[columns]
        self._df[self._weight_name()] = pd.np.zeros(len(self._df))
        self._check_targets(verbose)

    def dataframe(self, df, key_column=None):
        all_filter_cols = self._get_scheme_filter_cols()
        columns = self._columns(add_columns=[key_column])
        columns.extend(all_filter_cols)
        df = df.copy()[columns]
        df[self._weight_name()].replace(0, np.NaN, inplace=True)
        df.dropna(subset=[self._weight_name()], inplace=True)
        return df

    def _columns(self, identifier=None, add_columns=None):
        if identifier is not None:
            columns = [identifier]
        else:
            columns = []
        if add_columns:
            columns += add_columns
        [columns.append(target_col) for target_col in self.target_cols]
        columns.append(self._weight_name())
        return columns

    def _use_cap(self):
        if isinstance(self.cap, (list, tuple)):
            return True
        elif self.cap > 0:
            return True
        else:
            return False

    def group_targets(self, group_targets):
        """
        Set inter-group target proportions.

        This will scale the weight factors per group to match the desired group
        proportions and thus effectively change each group's weighted
        total number of cases.

        Parameters
        ----------
        group_targets : dict
            A dictionary mapping of group names to the desired proportions.

        Returns
        -------
        None
        """
        if isinstance(group_targets, dict):
            if all (group_targets[group] < 1 for group in group_targets):
                div_by = 1.0
            else:
                div_by = 100.0
            for group in group_targets:
                self._group_targets[group] = group_targets[group] / div_by
        else:
            raise ValueError(('Group_targets must be of type %s NOT %s ') % \
                             (type({}), type(group_targets)))

    def _weight_name(self):
        if self.weight_column_name is None:
            return self._WEIGHTS_ + self.name
        else:
            return self.weight_column_name

    def _empty_target_list(self):
        return {list_item: [] for list_item in self.target_cols}

    def _check_targets(self, verbose):
        """
        Check correct weight variable input proportion lengths and sum of 100.
        """

        some_nans = '*** Warning: Scheme "{0}", group "{1}" ***\n'\
                    'np.NaN found in weight variables:\n{2}\n'\
                    'Please check if weighted results are acceptable!\n'

        len_err_less = '*** Warning: Scheme "{0}", group "{1}" ***\nTargets for variable '\
                       '"{2}" do not match the number of sample codes.\n{3} codes '\
                       'expected, {4} codes found: Missing {5} in sample.\n'\
                       'Please check sample against scheme!\n'

        len_err_more = '*** Warning: Scheme "{0}", group "{1}" ***\nTargets for variable '\
                       '"{2}" do not match the number of sample codes.\n{3} codes '\
                       'expected, {4} codes found: Missing {5} in scheme.\n'\
                       'Please check sample against scheme!\n'

        sum_err = '*** Stopping: Scheme "{0}", group "{1}" ***\nThe targets for '\
                  'the variable "{2}" do not add up to 100.\nTarget sum is: {3}\n'

        vartype_err = '*** Stopping: Scheme "{0}", group "{1}" ***\n'\
                      'Variable "{2}" is unsuitable for Weighting.\n'\
                      'Target variables must be of type integer (convertable) / '\
                      'single categorical.\n'

        for group in self.groups:
            target_vars = [list(var.keys())[0] for var in
                           self.groups[group][self._TARGETS]]
            if self.groups[group][self._FILTER_DEF]:
                check_df = self._df.copy().query(
                    self.groups[group][self._FILTER_DEF]
                    )
            else:
                check_df = self._df.copy()
            nan_check = check_df[target_vars].isnull().sum()
            if not nan_check.sum() == 0:
                if verbose:
                    print(UserWarning(some_nans.format(
                        self.name, group, nan_check)))
            for target in self.groups[group][self._TARGETS]:
                target_col = list(target.keys())[0]
                target_codes = list(target.values())[0].keys()
                target_props = list(target.values())[0].values()
                sample_codes = check_df[target_col].value_counts(sort=False).index.tolist()

                miss_in_sample = [code for code in target_codes
                                  if code not in sample_codes
                                  and not list(target.values())[0][code] == 0.0]

                miss_in_targets = [code for code in sample_codes
                                   if code not in target_codes]

                if self._df[target_col].dtype == 'object':
                    raise ValueError(vartype_err.format(self.name, group, target_col))

                if miss_in_sample:
                    if verbose:
                        print(UserWarning(len_err_less.format(
                            self.name, group, target_col, len(target_codes),
                            len(sample_codes), miss_in_sample)))

                if miss_in_targets:
                    if verbose:
                        print(UserWarning(len_err_more.format(
                            self.name, group, target_col, len(target_codes),
                            len(sample_codes), miss_in_targets)))

                if not np.allclose(np.sum(list(target_props)), 100.0):
                    raise ValueError(sum_err.format(self.name, group,
                                    target_col, np.sum(target_props)))


    def validate(self):
        """
        Summary on scheme target variables to detect and handle missing data.

        Returns
        -------
        df : pandas.DataFrame
            A summary of missing entries and (rounded) mean/mode/median of
            value codes per target variable.
        """
        df = self._df.copy()[self.target_cols]
        nans = df.isnull().sum()
        means = np.round(df.mean(),0)
        modes = df.mode().iloc[0]
        medians = np.round(df.median(), 0)
        df = pd.concat([nans, means, modes, medians], axis=1)
        df.columns = ['missing', 'mean', 'mode', 'median']

        return df

class Rake:
    def __init__(self, dataframe, targets,
                 weight_column_name="weight",
                 max_iterations=1000,
                 convcrit=0.01,
                 _use_cap=False,
                 cap=10000000,
                 verbose=False,
                 anesrake_cap_correction=True):

        self.targets = targets
        self.dataframe = dataframe
        self.weight_column_name = weight_column_name

        self.cap = cap
        self.anesrake_cap_correction = anesrake_cap_correction
        self._use_cap = _use_cap
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
            raise Exception(
                "Unknown data type (%s). Should be <pandas.DataFrame>.",
                type(dataframe))
        self.pre_weight = pd.np.ones(len(self.dataframe))

        #Parse the targets
        self.rowcount = len(self.dataframe)
        col_names = [list(target.keys())[0] for target in targets]
        mappings = [{key: float(value) / 100 * self.rowcount for key, value
                     in list(target.values())[0].items()}
                    for target in targets]
        abs_targets = [{col_name: mapping} for col_name, mapping
                       in zip(col_names, mappings)]
        self.targets = abs_targets
        self.keys = col_names

        self.keys_row = self.keys[0:len(self.keys):2]
        self.keys_col = self.keys[1:len(self.keys):2]

        if pd.np.isnan(self.dataframe[self.weight_column_name]).sum() > 0:
            raise Exception("Seed weights cannot have missing values, use filter to eliminate missing values or substitute 1 for missing cases.")
        if cap <= 1 and _use_cap:
            raise Exception("Cap may not be less than or equal to 1.")
        if cap < 1.5 and _use_cap:
            print("Cap is very low, the model may take a long time to run.")

    def rakeonvar(self, target):
        target_col = list(target.keys())[0]
        for target_code, target_prop in list(target.values())[0].items():
            if target_prop == 0.00:
                target_prop = 0.00000001
            try:
                df = self.dataframe[(self.dataframe[target_col] == target_code)]
                index_array = (self.dataframe[target_col] == target_code)
                data = df[self.weight_column_name] * (target_prop / sum(df[self.weight_column_name]))
                self.dataframe.loc[index_array, self.weight_column_name] = data
            except:
               pass

    def calc_weight_efficiency(self):
        numerator = 100*sum(self.dataframe[self.weight_column_name] *
                            self.pre_weight) ** 2
        denominator = (sum(self.pre_weight) *
                       sum(self.pre_weight*
                           self.dataframe[self.weight_column_name] ** 2))
        self.weight_efficiency = numerator / denominator
        return self.weight_efficiency

    def generate_report(self):
        try:
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
        except MemoryError as e:
            warn = 'OOM: Could not finish writing report...'
            warnings.warn(warn)

    def start(self):
        pct_still = 1 - self.convcrit
        diff_error = 999999
        diff_error_old = 99999999999

        #cap (this needs more rigorous testings)
        if isinstance(self.cap, (list, tuple)):
            min_cap = self.cap[0]
            max_cap = self.cap[1]
        else:
            min_cap = None
            max_cap = self.cap

        if self.anesrake_cap_correction:
            max_cap += 0.0001
            if min_cap is not None:
                min_cap -= 0.0001

        for iteration in range(1, self.max_iterations+1):
            old_weights = self.dataframe[self.weight_column_name].copy()

            if not diff_error < pct_still * diff_error_old:
                break

            for target in self.targets:
                self.rakeonvar(target)

            if self._use_cap:

                if min_cap is None:
                    while self.dataframe[self.weight_column_name].max() > max_cap:
                        self.dataframe.loc[self.dataframe[self.weight_column_name] > max_cap, self.weight_column_name] = max_cap
                        self.dataframe[self.weight_column_name] = self.dataframe[self.weight_column_name]/pd.np.mean(self.dataframe[self.weight_column_name])
                else:
                    while (self.dataframe[self.weight_column_name].min() < min_cap) or (self.dataframe[self.weight_column_name].max() > max_cap):
                        self.dataframe.loc[self.dataframe[self.weight_column_name] < min_cap, self.weight_column_name] = min_cap
                        self.dataframe.loc[self.dataframe[self.weight_column_name] > max_cap, self.weight_column_name] = max_cap
                        self.dataframe[self.weight_column_name] = self.dataframe[self.weight_column_name]/pd.np.mean(self.dataframe[self.weight_column_name])

            diff_error_old = diff_error
            diff_error = sum(abs(self.dataframe[self.weight_column_name]-old_weights))

        self.iteration_counter = iteration  # for the report
        self.dataframe[self.weight_column_name].replace(0.00, 1.00, inplace=True)

        if iteration == self.max_iterations:
            print('Convergence did not occur in %s iterations' % iteration)
        else:
            if diff_error > 0.001:
                print("Raking achieved only partial convergence, please check the results to ensure that sufficient convergence was achieved.")
                print("No improvement was apparent after %s iterations" % iteration)
            else:
                if self.verbose:
                    print('Raking converged in %s iterations' % iteration)
                    print('Generating report')
        self.generate_report()
        return self.iteration_counter
