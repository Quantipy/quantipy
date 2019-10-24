
import pytest
from collections import OrderedDict
from quantipy.__imports__ import *  # noqa
from quantipy import (
    DataSet,
    Meta,
)

NAME = "Example Data (A)"
DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

@pytest.fixture(scope="class")
def dataset():
    name = NAME
    path = DATA
    ds = DataSet.from_quantipy(name, path)
    return ds

def _remove_files(name, suffix=[]):
    for suf in suffix:
        path = os.path.join(DATA, "{}.{}".format(name, suf))
        if os.path.exists(path):
            os.remove(path)

def _assert_exists(name, suffix=[]):
    for suf in suffix:
        path = os.path.join(DATA, "{}.{}".format(name, suf))
        assert os.path.exists(path)

def _assert_raise(caplog, err, msg, func, *args, **kwargs):
    caplog.clear()
    with pytest.raises(err):
        func(*args, **kwargs)
    assert caplog.records[-1].message == msg


class TestDataSet:
    # ------------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------------
    def test_aproperties(self, dataset):
        assert set(dataset.columns) == set([
            '@1', 'RecordNo', 'Wave', 'age', 'birth_day', 'birth_month',
            'birth_year', 'duration', 'end_time', 'ethnicity', 'gender',
            'locality', 'q1', 'q14r01c01', 'q14r01c02', 'q14r01c03',
            'q14r02c01', 'q14r02c02', 'q14r02c03', 'q14r03c01', 'q14r03c02',
            'q14r03c03', 'q14r04c01', 'q14r04c02', 'q14r04c03', 'q14r05c01',
            'q14r05c02', 'q14r05c03', 'q14r06c01', 'q14r06c02', 'q14r06c03',
            'q14r07c01', 'q14r07c02', 'q14r07c03', 'q14r08c01', 'q14r08c02',
            'q14r08c03', 'q14r09c01', 'q14r09c02', 'q14r09c03', 'q14r10c01',
            'q14r10c02', 'q14r10c03', 'q2', 'q2b', 'q3', 'q4', 'q5_1', 'q5_2',
            'q5_3', 'q5_4', 'q5_5', 'q5_6', 'q6_1', 'q6_2', 'q6_3', 'q7_1',
            'q7_2', 'q7_3', 'q7_4', 'q7_5', 'q7_6', 'q8', 'q8a', 'q9', 'q9a',
            'record_number', 'religion', 'start_time', 'unique_id', 'visit_1',
            'visit_2', 'visit_3', 'weight', 'weight_a', 'weight_b'])
        assert set(dataset.masks) == set([
            'q14_1', 'q14_2', 'q14_3', 'q5', 'q6', 'q7'])
        assert set(dataset.sets) == set([
            'data file', 'q14_1', 'q14_2', 'q14_3', 'q5', 'q6', 'q7'])
        assert set(dataset.singles) == set([
            'Wave', 'ethnicity', 'gender', 'locality', 'q1', 'q14r01c01',
            'q14r01c02', 'q14r01c03', 'q14r02c01', 'q14r02c02', 'q14r02c03',
            'q14r03c01', 'q14r03c02', 'q14r03c03', 'q14r04c01', 'q14r04c02',
            'q14r04c03', 'q14r05c01', 'q14r05c02', 'q14r05c03', 'q14r06c01',
            'q14r06c02', 'q14r06c03', 'q14r07c01', 'q14r07c02', 'q14r07c03',
            'q14r08c01', 'q14r08c02', 'q14r08c03', 'q14r09c01', 'q14r09c02',
            'q14r09c03', 'q14r10c01', 'q14r10c02', 'q14r10c03', 'q2b', 'q4',
            'q5_1', 'q5_2', 'q5_3', 'q5_4', 'q5_5', 'q5_6', 'q6_1', 'q6_2',
            'q6_3', 'q7_1', 'q7_2', 'q7_3', 'q7_4', 'q7_5', 'q7_6', 'religion',
            'visit_1', 'visit_2', 'visit_3'])
        assert set(dataset.delimited_sets) == set(['q2', 'q3', 'q8', 'q9'])
        assert set(dataset.ints) == set([
            '@1', 'RecordNo', 'age', 'birth_day', 'birth_month', 'birth_year',
            'record_number', 'unique_id'])
        assert set(dataset.floats) == set(['weight', 'weight_a', 'weight_b'])
        assert set(dataset.dates) == set(['end_time', 'start_time'])
        assert set(dataset.strings) == set(['q8a', 'q9a'])
        assert dataset.filters == []
        assert dataset.hidden_arrays == []

    # -------------------------------------------------------------------------
    # file i/o / conversions
    # -------------------------------------------------------------------------
    def test_from_quantipy(self, dataset):
        assert isinstance(dataset._meta, Meta)
        assert isinstance(dataset._data, pd.DataFrame)
        # io setup
        assert dataset.name == NAME
        assert dataset.path == DATA
        assert dataset.text_key == "en-GB"
        assert dataset.valid_tks == VALID_TKS
        assert dataset.dimensions_comp == False
        assert dataset.dimensions_suffix == "_grid"
        # data columns included in meta
        assert all(col in dataset.columns for col in dataset._data.columns)

    def test_to_quantipy(self, dataset):
        _remove_files("test_ds", ["json", "csv"])
        dataset.to_quantipy("test_ds")
        _assert_exists("test_ds", ["json", "csv"])
        _remove_files("test_ds", ["json", "csv"])

    def test_clone(self, dataset):
        ds = dataset.clone()
        assert isinstance(ds, DataSet)
        assert not ds == dataset
        # check meta
        assert isinstance(ds._meta, Meta)
        assert ds._meta == dataset._meta
        ds.text_key = "de-DE"
        assert not ds._meta == dataset._meta
        # check data
        assert isinstance(ds._data, pd.DataFrame)
        assert ds._data.equals(dataset._data)
        ds["@1"] = 2
        assert not ds._data.equals(dataset._data)

    def test_filter(self, dataset):
        ds = dataset.filter({"gender": 1})
        assert not ds == dataset
        # check meta
        assert isinstance(ds._meta, Meta)
        assert ds._meta == dataset._meta
        # check data
        assert ds._data.shape == (3952, 76)
        assert ds.get_codes_in_data("gender") == [1]

    def test_subset(self, dataset, caplog):
        ds = dataset.subset(["gender", "age", "q5"])
        assert ds._data.shape == (8255, 8)
        assert ds.variables_from_set() == ['age', 'gender', 'q5']

        msg = "Must either pass 'variables' or 'from_set'!"
        _assert_raise(caplog, ValueError, msg, dataset.subset)

        msg = "Must either pass 'variables' or 'from_set', not both!"
        _assert_raise(
            caplog, ValueError, msg, dataset.subset, ["gender"], "q5")

    # ------------------------------------------------------------------------
    # inspect
    # ------------------------------------------------------------------------
    def test_is_type(self, dataset, caplog):
        assert dataset.is_single("gender") == True
        assert dataset.is_delimited_set("gender") == False
        assert dataset.is_int("q5") == False
        assert dataset.is_single("q5") == True
        assert dataset.is_array("q5") == True
        assert dataset.is_float("age") == False
        assert dataset.is_date("start_time") == True
        assert dataset.is_array_item("q5_1") == True
        assert dataset.is_numeric("age") == True
        assert dataset.is_filter("gender") == False
        assert dataset.is_categorical("gender") == True

        msg = "'gender' is not of type string (but single)."
        _assert_raise(
            caplog, ValueError, msg, dataset.is_like_numeric, "gender")

    def test_empty(self, dataset):
        sources = dataset.get_sources("q5")
        assert dataset.empty("q5") == []
        assert dataset.empty("q5", {"age": is_le(15)}) == sources
        assert not dataset.empty("gender")
        assert dataset.empty("gender", {"age": is_le(15)})
        assert dataset.empty("weight")

    def test_all_any_nan(self, dataset):
        assert all(dataset.is_nan("weight"))
        assert len(dataset.any("q5", 5)) == 3907
        assert len(dataset.any("gender", 1)) == 3952
        assert len(dataset.all("q2", [1, 2])) == 800
        assert len(dataset.all("q5", 5)) == 102

    # ------------------------------------------------------------------------
    # add meta
    # ------------------------------------------------------------------------
    def test_add_meta_single(self, dataset):
        name, qtype, label = 'test', 'single', 'TEST VAR'
        cats1 = [(4, 'Cat1'), (5, 'Cat2')]
        cats2 = ['Cat1', 'Cat2']
        cats3 = [1, 2]
        for check, cat in enumerate([cats1, cats2, cats3], start=1):
            dataset.add_meta(name, qtype, label, cat)
            values = dataset.get_values(name)
            if check == 1:
                assert values == cats1
            elif check == 2:
                assert values == [(1, 'Cat1'), (2, 'Cat2')]
            elif check == 3:
                assert values == [(1, ''), (2, '')]

    def test_add_meta_array(self, dataset):
        name, qtype, label = 'array_test', 'delimited set', 'TEST LABEL TEXT'
        cats = ['Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5']
        items1 = [(1, 'ITEM A'), (3, 'ITEM B'), (6, 'ITEM C')]
        items2 = ['ITEM A', 'ITEM B', 'ITEM C']
        for check, items in enumerate([items1, items2], start=1):
            dataset.add_meta(name, qtype, label, cats, items)
            # check values and parent
            assert dataset.get_values(name) == list(enumerate(cats, start=1))
            value_ref = dataset._meta._get_value_ref(name)
            assert value_ref == "lib@values@array_test"
            assert dataset._meta["masks"][name]["values"] == value_ref
            sources = dataset.get_sources(name)
            parent = {'masks@array_test': {'type': 'array'}}
            for source in sources:
                assert dataset._meta["columns"][source]["values"] == value_ref
                assert dataset._meta["columns"][source]["parent"] == parent
            # check items
            items = dataset.get_items(name)
            if check == 1:
                assert items == [
                    ('array_test_1', 'ITEM A'),
                    ('array_test_3', 'ITEM B'),
                    ('array_test_6', 'ITEM C')]
            elif check == 2:
                assert items == [
                    ('array_test_1', 'ITEM A'),
                    ('array_test_2', 'ITEM B'),
                    ('array_test_3', 'ITEM C')]
            # check set
            data_file = dataset.get_set("data file")
            assert 'masks@array_test' in data_file
            assert 'columns@array_test_1' not in data_file

    def test_copy(self, dataset, caplog):
        # raises
        msg = "Must pass either 'copy_only' or 'copy_not', not both!"
        _assert_raise(
            caplog, ValueError, msg, dataset.copy,
            "gender", **{"copy_only": [1], "copy_not": [1]})

        msg = "Cannot copy a single array item."
        _assert_raise(caplog, ValueError, msg, dataset.copy, "q5_1")

        msg = "Cannot create 'Gender'. Weak duplicates exist: gender"
        _assert_raise(
            caplog, ValueError, msg, dataset.copy, "gender", "Gender")

        # full copy
        dataset.copy("q5")
        # check mask keys
        assert "q5" in dataset.masks
        assert "q5_rec" in dataset.masks
        # check set
        assert "masks@q5_rec" in dataset.get_set("data file")
        expect = ["columns@q5_rec_{}".format(x) for x in frange("1-6")]
        assert dataset.get_set("q5_rec") == expect
        # check sources
        expect = ["q5_rec_{}".format(x) for x in frange("1-6")]
        sources = dataset.get_sources("q5_rec")
        assert sources == expect
        # check values and parent
        ref1 = dataset._meta._get_value_ref("q5")
        ref2 = dataset._meta._get_value_ref("q5_rec")
        assert not ref1 == ref2
        assert dataset._meta[ref1] == dataset._meta[ref2]
        parent = {'masks@q5_rec': {'type': 'array'}}
        q5s = dataset.get_sources("q5")
        for s, source in zip(q5s, sources):
            assert dataset._meta["columns"][source]["values"] == ref2
            assert dataset._meta["columns"][source]["parent"] == parent
            assert dataset[s].equals(dataset[source])

        # slices copy
        dataset.copy("q5", "q5_rec2", True, {'gender': 1}, [1, 2, 3])
        assert dataset.get_codes("q5_rec2") == [1, 2, 3]
        assert not dataset.get_codes("q5") == [1, 2, 3]
        # data sliced and reduced properly?
        for source in dataset.get_sources('q5_rec2'):
            assert set(dataset[source].dropna().unique()) == set([1, 2, 3])
            assert dataset[[source, 'gender']].dropna()['gender'].unique() == 1

        # copy array data
        dataset.copy("q5", "q5_rec3", False)
        assert dataset.empty("q5_rec3") == dataset.get_sources("q5_rec3")
        dataset.copy_array_data("q5", "q5_rec3")

        msg = "Expect equal codes for source and target."
        _assert_raise(
            caplog, ValueError, msg, dataset.copy_array_data, "q6", "q5_rec3")

    # ------------------------------------------------------------------------
    # values
    # ------------------------------------------------------------------------
    def test_values_get(self, dataset, caplog):
        # values
        assert dataset.get_values("gender") == [(1, "Male"), (2, "Female")]

        msg = "Not categorical: 'q8a'"
        _assert_raise(caplog, TypeError, msg, dataset.get_values, "q8a")

        # value texts
        assert dataset.get_value_texts("gender") == ["Male", "Female"]

        assert dataset.get_codes_from_label("gender", "Male") == [1]
        assert dataset.get_codes_from_label(
            "gender", ["Male"], flat=False) == [1]

        # codes
        data = dataset._data.copy()
        dataset._data = data.head(20)
        assert dataset.get_codes("q8") == [1, 2, 3, 4, 5, 96, 98]
        assert dataset.get_codes_in_data("q8") == [1, 4, 5]
        dataset._data = data

    def test_values_modify(self, dataset, caplog):
        # extend values
        dataset.extend_values("q5", ["test", "new"])
        assert dataset.get_values("q5_1")[-2:] == [(99, "test"), (100, "new")]

        msg = "Cannot change values object of a single array item."
        _assert_raise(
            caplog, ValueError, msg, dataset.extend_values,
            "q5_1", ["test", "new"])
        msg = "Cannot add duplicated codes: [1]"
        _assert_raise(
            caplog, ValueError, msg, dataset.extend_values,
            "q5", [(1, "test")])
        msg = "Cannot add duplicated texts: ['test']"
        _assert_raise(
            caplog, ValueError, msg, dataset.extend_values,
            "q5", "test")

        # remove values
        dataset.remove_values("q5", [99, 100])
        assert len(dataset.get_codes("q5")) == 7

        msg = "Cannot change values object of a single array item."
        _assert_raise(
            caplog, ValueError, msg, dataset.remove_values, "q5_1", 1)

        msg = "Cannot remove all codes of a categorical variable."
        _assert_raise(
            caplog, ValueError, msg, dataset.remove_values,
            "q5", dataset.get_codes("q5"))

        # reorder values
        order = [98, 97, 5, 4, 3, 2, 1]
        dataset.reorder_values("q5", order)
        assert dataset.get_codes("q5_1") == order

        msg = "Cannot change values object of a single array item."
        _assert_raise(
            caplog, ValueError, msg, dataset.reorder_values, "q5_1", order)

        msg = "The new order does not take all variable codes into account"
        _assert_raise(
            caplog, ValueError, msg, dataset.reorder_values, "q5", [1, 2, 3])

    def test_factors(self, dataset):
        assert dataset.get_factors("gender") == OrderedDict()
        dataset.set_factors("gender", {2: 5})
        assert dataset.get_factors("gender") == OrderedDict([(2, 5)])
        dataset.del_factors("gender")
        assert dataset.get_factors("gender") == OrderedDict()

    # ------------------------------------------------------------------------
    # texts
    # ------------------------------------------------------------------------
    def test_texts_variables_get(self, dataset):

        text = dataset.get_text("gender")
        assert text == "What is your gender?"
        text = dataset.get_text("q6")
        assert text == "How often do you take part in any of the following?"
        texti = dataset.get_text("q6_1")
        assert texti == "Exercise alone"
        assert dataset.get_text("q6_1", False) == "{} - {}".format(text, texti)
        texts = [
            "Exercise alone",
            "Join an exercise class",
            "Play any kind of team sport"]
        assert dataset.get_item_texts("q6") == texts

    def test_texts_variables_set(self, dataset):
        dataset.set_text("q6", "new new", axis="x")
        assert dataset.get_text("q6", axis="x") == "new new"
        assert dataset.get_text("q6_1", False, axis="x").startswith("new new")
        dataset.set_text("gender", "GENDER", axis="x")
        assert dataset.get_text("gender", axis="x") == "GENDER"
        dataset.set_item_texts("q6", {1: "test"}, axis="x")
        assert dataset.get_text("q6_1", False, axis="x") == "new new - test"
        dataset.replace_texts({"new new": "new"})
        print (dataset._meta["columns"]["q6_1"]["text"])
        assert dataset.get_text("q6_1", False, axis="x") == "new - test"

    def test_text_keys(self, dataset, caplog):
        caplog.clear()
        with pytest.raises(ValueError):
            dataset.text_key = "bla"
        assert caplog.records[0].message == "'bla' is not a valid textkey!"
        assert dataset.text_key == "en-GB"
        assert dataset.used_text_keys() == ["en-GB"]
        dataset.force_texts("de-DE")
        assert set(dataset.used_text_keys()) == set(["en-GB", "de-DE"])
        dataset.select_text_keys("en-GB")
        assert dataset.used_text_keys() == ["en-GB"]
        gender = dataset._meta["columns"]["gender"]
        gender["text"]["x edits"] = "GENDER"
        dataset.repair_text_edits("en-GB")
        assert gender["text"]["x edits"] == {"en-GB": "GENDER"}

    # ------------------------------------------------------------------------
    # rules
    # ------------------------------------------------------------------------
    def test_sorting(self, dataset):
        dataset.set_sorting('q8', fix=[3, 98, 100])
        expect = {
            "sortx": {
                "fixed": [3, 98],
                "within": False,
                "between": False,
                "ascending": False,
                "sort_on": "@",
                "with_weight": "auto"}}
        assert dataset.get_rules("q8", "x") == expect

        dataset.set_sorting('q6', fix=["q6_1"], on="net_1")
        expect = {
            "sortx": {
                "fixed": ["q6_1"],
                "within": False,
                "between": False,
                "ascending": False,
                "sort_on": "net_1",
                "with_weight": "auto"}}
        assert dataset.get_rules("q6", "x") == expect

        dataset.set_sorting('q6', fix=[1, 2, 3], on="@")
        expect = {
            "sortx": {
                "fixed": [1, 2, 3],
                "within": False,
                "between": False,
                "ascending": False,
                "sort_on": "@",
                "with_weight": "auto"}}
        assert dataset.get_rules("q6_1", "x") == expect




    # def check_freq(self, dataset, var, show='values'):
    #     return freq(dataset._meta, dataset._data, var, show=show)

    # def check_cross(self, dataset, x, y, show='values', rules=False):
    #     return cross(dataset._meta, dataset._data, x=x, y=y,
    #                  show=show, rules=rules)

    # def test_order_full_change(self):
    #     dataset = self._get_dataset()
    #     variables = dataset._variables_from_set('data file')
    #     new_order = list(sorted(variables, key=lambda v: v.lower()))
    #     dataset.order(new_order)
    #     new_set_order = dataset._variables_to_set_format(new_order)
    #     data_file_items = dataset._meta['sets']['data file']['items']
    #     df_columns = dataset._data.columns.tolist()
    #     self.assertEqual(new_set_order, data_file_items)
    #     self.assertEqual(dataset.unroll(new_order), df_columns)

    # def test_order_repos_change(self):
    #     dataset = self._get_dataset()
    #     repos = [{'age': ['q8', 'q5']},
    #              {'q6': 'q7'},
    #              {'q5': 'weight_a'}]
    #     dataset.order(reposition=repos)
    #     data_file_items = dataset._meta['sets']['data file']['items']
    #     df_columns = dataset._data.columns.tolist()
    #     expected_items = ['record_number', 'unique_id', 'q8', 'weight_a', 'q5',
    #                       'age', 'birth_day', 'birth_month', 'birth_year',
    #                       'gender', 'locality', 'ethnicity', 'religion', 'q1',
    #                       'q2', 'q2b', 'q3', 'q4', 'q7', 'q6', 'q8a', 'q9',
    #                       'q9a', 'Wave', 'weight_b', 'start_time', 'end_time',
    #                       'duration', 'q14_1', 'q14_2', 'q14_3', 'RecordNo']
    #     expected_columns = dataset.unroll(expected_items)
    #     self.assertEqual(dataset._variables_to_set_format(expected_items),
    #                      data_file_items)
    #     self.assertEqual(expected_columns, df_columns)


    # def test_rename_via_masks(self):
    #     dataset = self._get_dataset()
    #     meta, data = dataset.split()
    #     new_name = 'q5_new'
    #     dataset.rename('q5', new_name)
    #     # name properly changend?
    #     self.assertTrue('q5' not in dataset.masks())
    #     self.assertTrue(new_name in dataset.masks())
    #     # item names updated?
    #     items = meta['sets'][new_name]['items']
    #     expected_items = ['columns@q5_new_1',
    #                       'columns@q5_new_2',
    #                       'columns@q5_new_3',
    #                       'columns@q5_new_4',
    #                       'columns@q5_new_5',
    #                       'columns@q5_new_6']
    #     self.assertEqual(items, expected_items)
    #     sources = dataset.sources(new_name)
    #     expected_sources = [i.split('@')[-1] for i in expected_items]
    #     self.assertEqual(sources, expected_sources)
    #     # lib reference properly updated?
    #     lib_ref_mask = meta['masks'][new_name]['values']
    #     lib_ref_items = meta['columns'][dataset.sources(new_name)[0]]['values']
    #     expected_lib_ref = 'lib@values@q5_new'
    #     self.assertEqual(lib_ref_mask, lib_ref_items)
    #     self.assertEqual(lib_ref_items, expected_lib_ref)
    #     # new parent entry correct?
    #     parent_spec = meta['columns'][dataset.sources(new_name)[0]]['parent']
    #     expected_parent_spec = {'masks@{}'.format(new_name): {'type': 'array'}}
    #     self.assertEqual(parent_spec, expected_parent_spec)
    #     # sets entries replaced?
    #     self.assertTrue('masks@q5' not in meta['sets']['data file']['items'])
    #     self.assertTrue('masks@q5_new' in meta['sets']['data file']['items'])
    #     self.assertTrue('q5' not in meta['sets'])
    #     self.assertTrue('q5_new' in meta['sets'])


    # def test_transpose(self):
    #     dataset = self._get_dataset(cases=500)
    #     meta, data = dataset.split()
    #     dataset.transpose('q5')
    #     # new items are old values?
    #     new_items = dataset.items('q5_trans')
    #     old_values = dataset.values('q5')
    #     check_old_values = [('q5_trans_{}'.format(element), text)
    #                         for element, text in old_values]
    #     self.assertEqual(check_old_values, new_items)
    #     # new values are former items?
    #     new_values = dataset.value_texts('q5_trans')
    #     old_items = dataset.item_texts('q5')
    #     self.assertEqual(new_values, old_items)
    #     # parent meta correctly updated?
    #     trans_parent = meta['columns'][dataset.sources('q5_trans')[0]]['parent']
    #     expected_parent = {'masks@q5_trans': {'type': 'array'}}
    #     self.assertEqual(trans_parent, expected_parent)
    #     # recoded data is correct?
    #     original_ct =  dataset.crosstab('q5', text=False)
    #     transposed_ct = dataset.crosstab('q5_trans', text=False)
    #     self.assertTrue(np.array_equal(original_ct.drop('All', 1, 1).T.values,
    #                     transposed_ct.drop('All', 1, 1).values))


    # def test_set_missings_flagging(self):
    #     dataset = self._get_dataset()
    #     dataset.set_missings('q8', {'exclude': [1, 2, 98, 96]})
    #     meta = dataset.meta('q8')[['codes', 'missing']]
    #     meta.index.name = None
    #     meta.columns.name = None
    #     missings = [[1, 'exclude'],
    #                 [2, 'exclude'],
    #                 [3, None],
    #                 [4, None],
    #                 [5, None],
    #                 [96, 'exclude'],
    #                 [98, 'exclude']]
    #     expected_meta = pd.DataFrame(missings,
    #                                  index=xrange(1, len(missings)+1),
    #                                  columns=['codes', 'missing'])
    #     self.assertTrue(all(meta == expected_meta))

    # def test_set_missings_results(self):
    #     dataset = self._get_dataset()
    #     dataset.set_missings('q8', {'exclude': [1, 2, 98, 96]})
    #     df = self.check_freq(dataset, 'q8')
    #     # check the base
    #     base_size = df.iloc[0, 0]
    #     expected_base_size = 1058
    #     self.assertEqual(base_size, expected_base_size)
    #     # check the index
    #     index = df.index.get_level_values(1).tolist()
    #     index.remove('All')
    #     expected_index = [3, 4, 5]
    #     self.assertEqual(index, expected_index)
    #     # check categories
    #     cat_vals = df.iloc[1:, 0].values.tolist()
    #     expected_cat_vals = [595, 970, 1235]
    #     self.assertEqual(cat_vals, expected_cat_vals)


    # def test_text_replacements_non_array(self):
    #     dataset = self._get_dataset()
    #     replace = {'following': 'TEST IN LABEL',
    #                'Breakfast': 'TEST IN VALUES'}
    #     dataset.replace_texts(replace=replace)
    #     expected_value = 'TEST IN VALUES'
    #     expected_label = 'Which of the TEST IN LABEL do you regularly skip?'
    #     value_text = dataset._get_valuemap('q8', non_mapped='texts')[0]
    #     column_text = dataset.text('q8')
    #     self.assertEqual(column_text, expected_label)
    #     self.assertEqual(value_text, expected_value)



    # def test_force_texts(self):
    #     dataset = self._get_dataset()
    #     dataset.set_value_texts(name='q4',
    #                             renamed_vals={1: 'kyllae'},
    #                             text_key='fi-FI')
    #     dataset.force_texts(copy_to='de-DE',
    #                         copy_from=['fi-FI','en-GB'],
    #                         update_existing=False)
    #     q4_de_val0 = dataset._meta['columns']['q4']['values'][0]['text']['de-DE']
    #     q4_de_val1 = dataset._meta['columns']['q4']['values'][1]['text']['de-DE']
    #     self.assertEqual(q4_de_val0, 'kyllae')
    #     self.assertEqual(q4_de_val1, 'No')

    #     q5_de_val0 = dataset._meta['lib']['values']['q5'][0]['text']['de-DE']
    #     self.assertEqual(q5_de_val0, 'I would refuse if asked')

    # def test_validate(self):
    #     dataset = self._get_dataset()
    #     meta = dataset._meta
    #     meta['columns']['q1']['values'][0]['text']['x edits'] = 'test'
    #     meta['columns']['q1']['name'] = 'Q1'
    #     meta['columns'].pop('q2')
    #     meta['masks']['q5']['text'] = {'en-GB': ''}
    #     meta['masks']['q6']['text'].pop('en-GB')
    #     meta['columns'].pop('q6_3')
    #     meta['columns']['q8']['text'] = ''
    #     meta['columns']['q8']['values'][3]['text'] = ''
    #     meta['columns']['q8']['values'] = meta['columns']['q8']['values'][0:5]
    #     index = ['q1', 'q2', 'q5', 'q6', 'q6_1', 'q6_2', 'q6_3', 'q8']
    #     data = {'name':     ['x', '',  '',  '',  '',  '',  '',  '' ],
    #             'q_label':  ['',  '',  'x', '',  '',  '',  '',  'x'],
    #             'values':   ['x', '',  '',  '',  '',  '',  '',  'x'],
    #             'text keys': ['',  '',  '',  'x', 'x', 'x', '',  'x'],
    #             'source':   ['',  '',  '',  'x', '',  '',  '',  '' ],
    #             'codes':    ['',  'x', '',  '',  '',  '',  'x', 'x']}
    #     df = pd.DataFrame(data, index=index)
    #     df = df[['name', 'q_label', 'values', 'text keys', 'source', 'codes']]
    #     df_validate = dataset.validate(False, verbose=False)
    #     self.assertTrue(df.equals(df_validate))

    # def test_compare(self):
    #     dataset = self._get_dataset()
    #     ds = dataset.clone()
    #     dataset.set_value_texts('q1', {2: 'test'})
    #     dataset.set_variable_text('q8', 'test', ['en-GB', 'sv-SE'])
    #     dataset.remove_values('q6', [1, 2])
    #     dataset.convert('q6_3', 'delimited set')
    #     index = ['q1', 'q6', 'q6_1', 'q6_2', 'q6_3', 'q8']
    #     data = {'type':         ['', '', '', '', 'x', ''],
    #             'q_label':      ['', '', '', '', '', 'en-GB, sv-SE, '],
    #             'codes':        ['', 'x', 'x', 'x', 'x', ''],
    #             'value texts': ['2: en-GB, ', '', '', '', '', '']}
    #     df = pd.DataFrame(data, index=index)
    #     df = df[['type', 'q_label', 'codes', 'value texts']]
    #     df_comp = dataset.compare(ds)
    #     self.assertTrue(df.equals(df_comp))

    # def test_uncode(self):
    #     dataset = self._get_dataset()
    #     dataset.uncode('q8',{1: 1, 2:2, 5:5}, 'q8', intersect={'gender':1})
    #     dataset.uncode('q8',{3: 3, 4:4, 98:98}, 'q8', intersect={'gender':2})
    #     df = dataset.crosstab('q8', 'gender')
    #     result = [[ 1797.,   810.,   987.],
    #               [  476.,     0.,   476.],
    #               [  104.,     0.,   104.],
    #               [  293.,   293.,     0.],
    #               [  507.,   507.,     0.],
    #               [  599.,     0.,   599.],
    #               [  283.,   165.,   118.],
    #               [   26.,    26.,     0.]]
    #     self.assertEqual(df.values.tolist(), result)

    # def test_derotate_df(self):
    #     dataset = self._get_dataset()
    #     levels = {'visit': ['visit_1', 'visit_2', 'visit_3']}
    #     mapper = [{'q14r{:02}'.format(r): ['q14r{0:02}c{1:02}'.format(r, c)
    #               for c in range(1, 4)]} for r in frange('1-5')]
    #     ds = dataset.derotate(levels, mapper, 'gender', 'record_number')
    #     df_h = ds._data.head(10)
    #     df_val = [[x if not np.isnan(x) else 'nan' for x in line]
    #               for line in df_h.values.tolist()]
    #     result_df = [[1.0, 2.0, 1.0, 4.0, 4.0, 4.0, 8.0, 1.0, 2.0, 4.0, 2.0, 3.0, 1.0],
    #                  [1.0, 2.0, 2.0, 4.0, 4.0, 4.0, 8.0, 3.0, 3.0, 2.0, 4.0, 3.0, 1.0],
    #                  [1.0, 3.0, 1.0, 1.0, 1.0, 8.0, 'nan', 4.0, 3.0, 1.0, 3.0, 1.0, 2.0],
    #                  [1.0, 4.0, 1.0, 5.0, 5.0, 4.0, 8.0, 2.0, 3.0, 2.0, 3.0, 1.0, 1.0],
    #                  [1.0, 4.0, 2.0, 4.0, 5.0, 4.0, 8.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0],
    #                  [1.0, 5.0, 1.0, 3.0, 3.0, 5.0, 8.0, 4.0, 2.0, 2.0, 1.0, 3.0, 1.0],
    #                  [1.0, 5.0, 2.0, 5.0, 3.0, 5.0, 8.0, 3.0, 3.0, 3.0, 1.0, 2.0, 1.0],
    #                  [1.0, 6.0, 1.0, 2.0, 2.0, 8.0, 'nan', 4.0, 2.0, 3.0, 4.0, 2.0, 1.0],
    #                  [1.0, 7.0, 1.0, 3.0, 3.0, 3.0, 8.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0],
    #                  [1.0, 7.0, 2.0, 3.0, 3.0, 3.0, 8.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0]]
    #     result_columns = ['@1', 'record_number', 'visit', 'visit_levelled',
    #                       'visit_1', 'visit_2', 'visit_3', 'q14r01', 'q14r02',
    #                       'q14r03', 'q14r04', 'q14r05', 'gender']
    #     df_len = 18520
    #     self.assertEqual(df_val, result_df)
    #     self.assertEqual(df_h.columns.tolist(), result_columns)
    #     self.assertEqual(len(ds._data.index), df_len)
    #     path_json = '{}/{}.json'.format(ds.path, ds.name)
    #     path_csv = '{}/{}.csv'.format(ds.path, ds.name)
    #     os.remove(path_json)
    #     os.remove(path_csv)

    # def test_derotate_freq(self):
    #     dataset = self._get_dataset()
    #     levels = {'visit': ['visit_1', 'visit_2', 'visit_3']}
    #     mapper = [{'q14r{:02}'.format(r): ['q14r{0:02}c{1:02}'.format(r, c)
    #               for c in range(1, 4)]} for r in frange('1-5')]
    #     ds = dataset.derotate(levels, mapper, 'gender', 'record_number')
    #     val_c = {'visit': {'val': {1: 8255, 2: 6174, 3: 4091},
    #                'index': [1, 2, 3]},
    #              'visit_levelled': {'val': {4: 3164, 1: 3105, 5: 3094, 6: 3093, 3: 3082, 2: 2982},
    #                                'index': [4, 1, 5, 6, 3,2]},
    #              'visit_1': {'val': {4: 3225, 6: 3136, 3: 3081, 2: 3069, 1: 3029, 5: 2980},
    #                          'index': [4, 6, 3, 2, 1, 5]},
    #              'visit_2': {'val': {1: 2789, 6: 2775, 5: 2765, 3: 2736, 4: 2709, 2: 2665, 8: 2081},
    #                          'index': [1, 6, 5, 3, 4, 2, 8]},
    #              'visit_3': {'val': {8: 4166, 5: 2181, 4: 2112, 3: 2067, 1: 2040, 6: 2001, 2: 1872},
    #                          'index': [8, 5, 4, 3, 1, 6, 2]},
    #              'q14r01': {'val': {3: 4683, 1: 4653, 4: 4638, 2: 4546},
    #                         'index': [3, 1, 4, 2]},
    #              'q14r02': {'val': {4: 4749, 2: 4622, 1: 4598, 3: 4551},
    #                         'index': [4, 2, 1, 3]},
    #              'q14r03': {'val': {1: 4778, 4: 4643, 3: 4571, 2: 4528},
    #                         'index': [1, 4, 3, 2]},
    #              'q14r04': {'val': {1: 4665, 2: 4658, 4: 4635, 3: 4562},
    #                         'index': [1, 2, 4, 3]},
    #              'q14r05': {'val': {2: 4670, 4: 4642, 1: 4607, 3: 4601},
    #                        'index': [2, 4, 1, 3]},
    #              'gender': {'val': {2: 9637, 1: 8883},
    #                         'index': [2, 1]}}
    #     for var in val_c.keys():
    #         series = pd.Series(val_c[var]['val'], index = val_c[var]['index'])
    #         compare = all(series == ds._data[var].value_counts())
    #         self.assertTrue(compare)
    #     path_json = '{}/{}.json'.format(ds.path, ds.name)
    #     path_csv = '{}/{}.csv'.format(ds.path, ds.name)
    #     os.remove(path_json)
    #     os.remove(path_csv)

    # def test_derotate_meta(self):
    #     dataset = self._get_dataset()
    #     levels = {'visit': ['visit_1', 'visit_2', 'visit_3']}
    #     mapper = [{'q14r{:02}'.format(r): ['q14r{0:02}c{1:02}'.format(r, c)
    #               for c in range(1, 4)]} for r in frange('1-5')]
    #     ds = dataset.derotate(levels, mapper, 'gender', 'record_number')
    #     err = ds.validate(False, False)
    #     err_s = None
    #     self.assertEqual(err_s, err)
    #     path_json = '{}/{}.json'.format(ds.path, ds.name)
    #     path_csv = '{}/{}.csv'.format(ds.path, ds.name)
    #     os.remove(path_json)
    #     os.remove(path_csv)

    # def test_interlock(self):
    #     dataset = self._get_dataset()
    #     data = dataset._data
    #     name, lab = 'q4AgeGen', 'q4 Age Gender'
    #     variables = ['q4',
    #                  {'age': [(1, '18-35', {'age': frange('18-35')}),
    #                           (2, '30-49', {'age': frange('30-49')}),
    #                           (3, '50+', {'age': is_ge(50)})]},
    #                  'gender']
    #     dataset.interlock(name, lab, variables)
    #     val = [1367,1109,1036,831,736,579,571,550,454,438,340,244]
    #     ind = ['10;','8;','9;','7;','3;','8;10;','1;','4;','2;','7;9;','1;3;','2;4;']
    #     s = pd.Series(val, index=ind, name='q4AgeGen')
    #     self.assertTrue(all(s==data['q4AgeGen'].value_counts()))
    #     values = [(1, u'Yes/18-35/Male'),
    #               (2, u'Yes/18-35/Female'),
    #               (3, u'Yes/30-49/Male'),
    #               (4, u'Yes/30-49/Female'),
    #               (5, u'Yes/50+/Male'),
    #               (6, u'Yes/50+/Female'),
    #               (7, u'No/18-35/Male'),
    #               (8, u'No/18-35/Female'),
    #               (9, u'No/30-49/Male'),
    #               (10, u'No/30-49/Female'),
    #               (11, u'No/50+/Male'),
    #               (12, u'No/50+/Female')]
    #     text = 'q4 Age Gender'
    #     self.assertEqual(values, dataset.values('q4AgeGen'))
    #     self.assertEqual(text, dataset.text('q4AgeGen'))
    #     self.assertTrue(dataset.is_delimited_set('q4AgeGen'))

    # def test_dichotomous_to_delimited_set(self):
    #     dataset = self._get_dataset()
    #     dataset.dichotomize('q8', None, False)
    #     dataset.to_delimited_set('q8_new', dataset.text('q8'),
    #                              ['q8_1', 'q8_2', 'q8_3', 'q8_4', 'q8_5', 'q8_96', 'q8_98'],
    #                              from_dichotomous=True, codes_from_name=True)
    #     self.assertEqual(dataset.values('q8'), dataset.values('q8_new'))
    #     self.assertEqual(dataset['q8'].value_counts().values.tolist(),
    #                      dataset['q8_new'].value_counts().values.tolist())
    #     self.assertRaises(ValueError, dataset.to_delimited_set, 'q8_new', '', ['age', 'gender'])

    # def test_categorical_to_delimited_set(self):
    #     dataset = self._get_dataset()
    #     self.assertRaises(ValueError, dataset.to_delimited_set, 'q1_1', '', ['q1', 'q2'])
    #     dataset.to_delimited_set('q5_new',
    #                      dataset.text('q5'),
    #                      dataset.sources('q5'),
    #                      False)
    #     self.assertEqual(dataset.crosstab('q5_new').values.tolist(),
    #                      [[8255.0], [3185.0], [2546.0], [4907.0],
    #                       [287.0], [3907.0], [1005.0], [3640.0]])
    #     for v in dataset.sources('q5'):
    #         self.assertEqual(dataset.values('q5_new'), dataset.values(v))

    # def test_get_value_texts(self):
    #     dataset = self._get_dataset()
    #     values = [(1, u'Regularly'), (2, u'Irregularly'), (3, u'Never')]
    #     self.assertEqual(values, dataset.values('q2b', 'en-GB'))
    #     dataset._meta['columns']['q2b']['values'][0]['text']['x edits'] = {'en-GB': 'test'}
    #     value_texts = ['test', None, None]
    #     self.assertEqual(value_texts, dataset.value_texts('q2b', 'en-GB', 'x'))

    # def test_get_item_texts(self):
    #     dataset = self._get_dataset()
    #     items = [(u'q6_1', u'Exercise alone'),
    #              (u'q6_2', u'Join an exercise class'),
    #              (u'q6_3', u'Play any kind of team sport')]
    #     self.assertEqual(items, dataset.items('q6', 'en-GB'))
    #     dataset._meta['masks']['q6']['items'][2]['text']['x edits'] = {'en-GB': 'test'}
    #     item_texts = [None, None, 'test']
    #     self.assertEqual(item_texts, dataset.item_texts('q6', 'en-GB', 'x'))

    # def test_get_variable_text(self):
    #     dataset = self._get_dataset()
    #     text = 'How often do you take part in any of the following? - Exercise alone'
    #     self.assertEqual(text, dataset.text('q6_1', False, 'en-GB'))
    #     text = 'Exercise alone'
    #     self.assertEqual(text, dataset.text('q6_1', True, 'en-GB'))
    #     text = ''
    #     self.assertEqual(text, dataset.text('q6_1', True, 'en-GB', 'x'))

    # def test_set_value_texts(self):
    #     dataset = self._get_dataset()
    #     values = [{u'text': {u'en-GB': u'Strongly disagree'}, u'value': 1},
    #               {u'text': {u'en-GB': 'test1'}, u'value': 2},
    #               {u'text': {u'en-GB': u'Neither agree nor disagree'}, u'value': 3},
    #               {u'text': {u'en-GB': u'Agree', 'y edits': {'en-GB': 'test2'}}, u'value': 4},
    #               {u'text': {u'en-GB': u'Strongly agree'}, u'value': 5}]
    #     dataset.set_value_texts('q14_1', {2: 'test1'}, 'en-GB')
    #     dataset.set_value_texts('q14_1', {4: 'test2'}, 'en-GB', 'y')
    #     value_obj = dataset._meta['lib']['values']['q14_1']
    #     self.assertEqual(value_obj, values)
    #     values = [{u'text': {u'en-GB': u'test1'}, u'value': 1},
    #               {u'text': {u'en-GB': u'Irregularly'}, u'value': 2},
    #               {u'text': {u'en-GB': u'Never',
    #                          u'y edits': {'en-GB': 'test2'},
    #                          u'x edits': {'en-GB': 'test2'}}, u'value': 3}]
    #     dataset.set_value_texts('q2b', {1: 'test1'}, 'en-GB')
    #     dataset.set_value_texts('q2b', {3: 'test2'}, 'en-GB', ['x', 'y'])
    #     value_obj = dataset._meta['columns']['q2b']['values']
    #     self.assertEqual(value_obj, values)

    # def test_set_item_texts(self):
    #     dataset = self._get_dataset()
    #     items = [{u'en-GB': u'Exercise alone'},
    #              {u'en-GB': u'Join an exercise class',
    #               'sv-SE': 'test1',
    #               'x edits': {'sv-SE': 'test', 'en-GB': 'test'}},
    #              {u'en-GB': u'Play any kind of team sport',
    #               'sv-SE': 'test2'}]
    #     dataset.set_item_texts('q6', {2: 'test1', 3: 'test2'}, 'sv-SE')
    #     dataset.set_item_texts('q6', {2: 'test'}, ['en-GB', 'sv-SE'], 'x')
    #     item_obj = [i['text'] for i in dataset._meta['masks']['q6']['items']]
    #     self.assertEqual(item_obj, items)

    # def test_set_variable_text(self):
    #     dataset = self._get_dataset()
    #     text = {'en-GB': 'new text', 'sv-SE': 'new text'}
    #     dataset.set_variable_text('q6', 'new text', ['en-GB', 'sv-SE'])
    #     dataset.set_variable_text('q6', 'new', ['da-DK'], 'x')
    #     text_obj = dataset._meta['masks']['q6']['text']
    #     self.assertEqual(text_obj, text)
    #     text = {'en-GB': 'What is your main fitness activity?',
    #             'x edits': {'en-GB': 'edit'}, 'y edits':{'en-GB': 'edit'}}
    #     dataset.set_variable_text('q1', 'edit', 'en-GB', ['x', 'y'])

    # def test_crosstab(self):
    #     x = 'q14r01c01'
    #     dataset = self._get_dataset()
    #     dataset.crosstab(x)
    #     self.assertEqual(dataset._meta['columns'][x]['values'],
    #                      'lib@values@q14_1')
