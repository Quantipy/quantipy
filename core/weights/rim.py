import io
import re
import pandas as pd
import itertools
import pdb
import pandas as pd

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
        self.lists = lists
        self.max_iterations = 1000
        self.convcrit = 0.01
        self.cap = 0
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
        self.__FILTERS = 'filters'
        self.__TARGETS = 'targets'
        self.__TARGETS_INDEX = 'targets_index'
        self.__REPORT = 'report'
        self.__DEFAULT_GROUP_NAME = '__default_group_name__'
        self.__WEIGHTS_ = 'weights_'

        # default group init
        # a group can have any name except for the __DEFAULT_GROUP_NAME
        # __DEFAULT_GROUP_NAME is used when no group is defined for the weight scheme
        self.groups = {}
        self.groups[self.__DEFAULT_GROUP_NAME] = {}
        self.groups[self.__DEFAULT_GROUP_NAME][self.__REPORT] = None
        self.groups[self.__DEFAULT_GROUP_NAME][self.__FILTERS] = None
        self.groups[self.__DEFAULT_GROUP_NAME][self.__TARGETS] = self.__generate_empty_target_lists()
        self.groups[self.__DEFAULT_GROUP_NAME][self.__TARGETS_INDEX] = None

        self.uses_default_group = True   # This is changed if a group is added

    def report(self, group=None):
        report = {}
        if group is None:
            for group in self.groups:
                report[group] = self.groups[group][self.__REPORT]
        else:
            report[group] = self.groups[group][self.__REPORT]
        return report

    def impute_method(self, method, target=None):
        if target is None:
            self._impute_method = method
        else:
            self._impute_method_specific[target] = method

    def generate(self):
        for group in self.groups:
            filters = self.groups[group][self.__FILTERS]
            targets = self.groups[group][self.__TARGETS]
            if filters is None:
                sub_df = self._df.copy()
            else:
                sub_df = self._df[self.__create_index_series(self._df, filters)].copy()
            rake = Rake(sub_df, targets, self.weight_name())
            rake.start()
            self.groups[group][self.__REPORT] = rake.report
            self._df.loc[rake.dataframe.index, self.weight_name()] = rake.dataframe[self.weight_name()]
        if self.total > 0:
            self._df[self.weight_name()].replace(1,0,inplace=True)
            totalw =  self._df[self.weight_name()].sum()
            self._df[self.weight_name()] = self._df[self.weight_name()] / self._df[self.weight_name()].sum() * self.total
            filter = self.groups[group][self.__FILTERS].keys()[0]
            values = self.groups[group][self.__FILTERS].values()[0]
            for group in self.groups:
                self.groups[group]['report']['summary']['Total: weighted'] = self._df.loc[self._df[filter] == values, self.weight_name()].sum()
        for group in self._group_targets:
            filter = self.groups[group][self.__FILTERS].keys()[0]
            values = self.groups[group][self.__FILTERS].values()[0]
            sub_weight_sum = self._df[self.weight_name()][self._df[filter] == values].sum()
            if self.total == 0: 
                self.total =  sum(
                    [len(self._df[self._df[key] == value].index)
                     for group_target in self._group_targets
                     for key, value in self.groups[group_target][self.__FILTERS].iteritems()]
                )                    
            ratio = self._group_targets[group] / (sub_weight_sum / self.total)
            self._df.loc[self._df[filter] == values, self.weight_name()] = self._df[self.weight_name()] * ratio
            self.groups[group]['report']['summary']['Total: weighted'] = self._df.loc[self._df[filter] == values, self.weight_name()].sum()

        return self._df[self.weight_name()]

    def minimize_columns(self, df, key):
        # Add the columns from the filters as well as the lists
        group_filter_columns = []
        for group in self.groups:
            if isinstance(self.groups[group]['filters'], dict):
                filter = self.groups[group]['filters'].keys()[0]
                if group_filter_columns.count(filter) == 0:
                    group_filter_columns.append(filter)

        columns = [key] + self.lists + group_filter_columns

        self._df = pd.DataFrame(df, columns=columns, copy=True)
        self._df[self.weight_name()] = pd.np.ones(len(self._df))
        self.__dropna()

        # Check if the targets are of the right number (correct number of targets)
        self.__check_targets()  # This function throws an error if the targets are incorrect.

    def dataframe(self, df, index=None, key_column=None):
        columns = self.columns(add_columns=[key_column])
        filters = self.filters()

        # if index is None:
        #     columns.append(index)
        index = self.__create_index_series(df, filters)
        return df.ix[index, columns + filters.keys()]

    def set_targets(self, targets, group_name=None):

        group_name = self.__DEFAULT_GROUP_NAME if group_name is None else group_name

        if not isinstance(targets, dict):
            raise ValueError(("'targets' should be type dict, '%s' was given.") % type(targets))
            return False
        else:
            self.groups[group_name][self.__TARGETS] = {}
            for list_name, values in targets.iteritems():
                if not isinstance(values, list):
                    raise ValueError("'%s's targets should be type list, '%s' was given." % (list_name, type(targets)))
                    return False
                else:
					if not list_name in self.lists:
					    self.lists.append(list_name)
					self.groups[group_name][self.__TARGETS][list_name] = values

    def group_filter(self, group_name=None, filter=None):
        if group_name is not None and filter is not None:
            if isinstance(filter, str):
                filter = self.__convert_filter_string(filter)
                if filter is False:
                    return False  # Filter is in an incorrect format
            if isinstance(filter, dict):
                if group_name in self.groups:
                    self.groups[group_name][self.__FILTERS] = filter
                else:
                    raise ValueError(("The group-name '%s' is not in the available groups.\nAvailable groups are %s.") % (group_name, self.groups.keys()))
            else:
                raise ValueError("Unknown filter type, should be string format or a dictionary")
                return False
        else:
            return False

    # The filter parameter can be a string in this format 'column==value'
    # or dict in this format {'column':value} (where value is a number)
    def add_group(self, name=None, filter=None, targets=None):
        name = self.__DEFAULT_GROUP_NAME if name is None else name  # James changes

        # If the group does not exist then create it
        if name not in self.groups:
            self.groups[name] = {}

        # generate an empty target list if it does not exist
        if self.__TARGETS not in self.groups[name]:
            self.groups[name][self.__TARGETS] = self.__generate_empty_target_lists()

        # Delete the default group if it exists AND there are other groups
        if self.uses_default_group and len(self.groups.keys()) > 1:
            self.uses_default_group = False
            del self.groups[self.__DEFAULT_GROUP_NAME]

        if isinstance(filter, str):
            filter = self.__convert_filter_string(filter)
            if filter is False:
                return False  # Filter is in an incorrect format

            # TODO :: Check if the filter key is in the columns and that the value is valid
            # filter.keys[0] in self.dataframe.columns

        self.groups[name][self.__FILTERS] = filter
        if targets is not None:
            self.set_targets(targets=targets, group_name=name)

    def rename_list(self, find=None, replace=None):
        if find is not None and replace is not None:
            try:
                index = self.lists.index(find)
                self.lists[index] = replace
                return True
            except ValueError:
                raise ValueError(("Could not find '%s' in %s.") % (find, self.lists))
                return False
        else:
            return False

    def use_cap(self):
        if self.cap <= 0:
            return False
        else:
            return True

    def filters(self):
        filter_keys = [self.groups[group_key][self.__FILTERS] for group_key in self.groups]
        if None in filter_keys: filter_keys.remove(None)
        filters = {}
        for group_filter in filter_keys:
            for key in group_filter:
                if key not in filters:
                    filters[key] = [int(group_filter[key])]
                else:
                    filters[key].append(int(group_filter[key]))

        return filters

    def group_targets(self, group_targets):
        if isinstance(group_targets, dict):
            for group in group_targets:
                if group_targets[group] < 1:
                    self._group_targets[group] = group_targets[group]
                else:
                    self._group_targets[group] = group_targets[group] / 100.0
        else:
            raise ValueError(('Group_targets must be of type %s NOT %s ') % (type({}), type(group_targets)))

    def columns(self, identifier=None, add_columns=None):
        if identifier is not None:
            columns = [identifier]
        else:
            columns = []
        if add_columns:
            columns += add_columns
        [columns.append(list_item) for list_item in self.lists]
        columns.append(self.weight_name())
        return columns

    def weight_name(self):
        if self.weight_column_name is None:
            return self.__WEIGHTS_ + self.name
        else:
            return self.weight_column_name

    def __dataframe_subset(self, group):
        pass

    def __generate_empty_target_lists(self):
        return {list_item: [] for list_item in self.lists}

    def __create_index_series(self, df, filters):
        # Create a True/False index serie
        index = None
        if not filters or filters is None:
            index = df.index  # The entire dataset
        else:
            for key in filters:
                if isinstance(filters[key], list):
                    tmp_index = df[key].isin(filters[key])
                else:
                    tmp_index = df[key].isin([int(filters[key])])

                if index is None:
                    index = tmp_index
                else:
                    index = pd.np.logical_and(index, tmp_index)
        return index

    def __convert_filter_string(self, filter):
        # The string has to have a string on the left x of an equal sign and a number on the right x
        # regexp = re.compile(r'[a-zA-Z0-9]=[a-zA-Z0-9]')
        regexp = re.compile(r'^.+={1,2}[0-9]+$')

        if regexp.match(filter):
            filter = filter.split('=')
            filter = {filter[0]: int(filter[len(filter)-1])}
            return filter
        else:
            raise Exception('The string has to have a string on the left x of an equal sign and a number on the right x')
            return False

    def __dropna(self):
        if self.dropna:
            self._df.dropna(how='any', inplace=True)
        else:
            columns = self._impute_method_specific
            columns.update({column: self._impute_method for column in self.lists if column not in self._impute_method_specific.keys()})

            for column, method in columns.iteritems():
                if method == "mean":
                    self._df[column].fillna(self._df[column].mean(), inplace=True)
                elif method == "mode":
                    self._df[column].fillna(self._df[column].mode()[0], inplace=True)

    def __check_targets(self):
        """ Verify that there are a correct ammount of targets

        This function returns true.
        This function raises an error if any of the checks fail
        """
        error_message = 'The targets for the variable "{0}" in scheme "{1}" do not match the dataframes unique values. It should have been {2}, got {3}.'

        # target_values looks like this => {'target_column':[1, 2, 3, 4, 5, 6], ... }
        target_values = {}
        for target_column in self._df[self.lists]:
            unique_sorted_list = pd.unique(self._df[target_column])
            unique_sorted_list.sort()
            target_values[target_column] = unique_sorted_list.tolist()

        for group in self.groups:
            for target_column, target_ratios in self.groups[group][self.__TARGETS].iteritems():
                assert len(target_ratios) == len(target_values[target_column]), error_message.format(target_column, self.name, len(target_values[target_column]), len(target_ratios))
            if self.__TARGETS_INDEX in self.groups[group] and self.groups[group][self.__TARGETS_INDEX] is not None:
                for target_column, target_ratios in self.groups[group][self.__TARGETS_INDEX].iteritems():
                    assert target_ratios == target_values[target_column], error_message.format(target_column, self.name, target_values[target_column], target_ratios)

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
