import pdb;
import unittest
import os.path
import io
import test_helper
import pandas as pd

from quantipy.core.stack import Stack
from quantipy.core.chain import Chain
from quantipy.core.cluster import Cluster
from quantipy.core.view_generators.view_maps import QuantipyViews
from quantipy.core.helpers.functions import load_json

CBASE = "x|frequency|x:y|||cbase"
COUNTS = "x|frequency||||counts"

class TestClusterObject(unittest.TestCase):

    def setUp(self):
        self.path = './tests/'
#         self.path = ''
        self.project_name = 'Example Data (A)'
        
        # Load Example Data (A) data and meta into self
        name_data = '%s.csv' % (self.project_name)
        path_data = '%s%s' % (self.path, name_data)
        self.example_data_A_data = pd.DataFrame.from_csv(path_data)        
        name_meta = '%s.json' % (self.project_name)
        path_meta = '%s%s' % (self.path, name_meta)
        self.example_data_A_meta = load_json(path_meta)

        # Variables by type for Example Data A
        self.int = ['record_number', 'unique_id', 'age', 'birth_day', 'birth_month']
        self.float = ['weight', 'weight_a', 'weight_b']
        self.single = ['gender', 'locality', 'ethnicity', 'religion', 'q1']
        self.delimited_set = ['q2', 'q3', 'q8', 'q9']
        self.string = ['q8a', 'q9a']
        self.date = ['start_time', 'end_time']
        self.time = ['duration']
        self.array = ['q5', 'q6', 'q7']
        
        # The minimum list of variables required to populate a stack with all single*delimited set variations
        self.minimum = ['q2b', 'Wave', 'q2', 'q3', 'q5_1']
        self.one_of_each = ['record_number', 'weight', 'gender', 'q2', 'q8a', 'start_time', 'duration']

        # Set up example stacks
        self.stack0 = self.setup_stack_Example_Data_A(name='Jan')
        self.stack1 = self.setup_stack_Example_Data_A(name='Feb')
        self.stack2 = self.setup_stack_Example_Data_A(name='Mar')
        self.stack3 = self.setup_stack_Example_Data_A(name='Apr')

        self.path_cluster = '%sClusterName.cluster' % (self.path)

        if os.path.exists(self.path_cluster):
            os.remove(self.path_cluster)

    def test_add_chain_exceptions(self):

        y = self.minimum[0]
        x = self.minimum[1:3]

        cluster = Cluster(name="ClusterName")
        self.assertIsInstance(cluster, Cluster)
        self.assertItemsEqual([],cluster.keys())

        exception_message = "You must pass either a Chain or a list of Chains to Cluster.add_chain()"

        # Test the exceptions in add_chain
        with self.assertRaises(TypeError) as cm:
            cluster.add_chain()
        self.assertEquals(cm.exception.message, exception_message)

        with self.assertRaises(TypeError) as cm:
            cluster.add_chain(chains=None)
        self.assertEquals(cm.exception.message, exception_message)

        with self.assertRaises(TypeError) as cm:
            cluster.add_chain(chains="This is not a chain")
        self.assertEquals(cm.exception.message, exception_message)

        exception_message = "One or more of the supplied chains has an inappropriate type."

        chain_1 = self.stack0.get_chain(name="ChainName1", data_keys="Jan", x=x, y=y, views=[COUNTS])
        chain_2 = self.stack1.get_chain(name="ChainName2", data_keys="Feb", x=x, y=y, views=[COUNTS])

        with self.assertRaises(TypeError) as cm:
            cluster.add_chain(chains=[chain_1, "This is not a chain", chain_2])
        self.assertEquals(cm.exception.message, exception_message)

        # Should succeed
        try:
            cluster.add_chain(chains=chain_1)
        except TypeError as cm:
            self.fail('cluster.add_chain(chains=chain_1) should NOT raise a TypeError Exception.')
        except:
            pass

    def test_add_chain(self):
        y = self.minimum[0]
        x = self.minimum[1:3]

        cluster = Cluster(name="ClusterName")
        self.assertIsInstance(cluster, Cluster)
        self.assertItemsEqual([],cluster.keys())

        chain_1 = self.stack0.get_chain(name="ChainName1", data_keys="Jan", x=x, y=y, views=[COUNTS])
        chain_2 = self.stack1.get_chain(name="ChainName2", data_keys="Feb", x=x, y=y, views=[COUNTS])
        chain_3 = self.stack2.get_chain(name="ChainName3", data_keys="Mar", x=x, y=y, views=[COUNTS])
        chain_4 = self.stack3.get_chain(name="ChainName4", data_keys="Apr", x=x, y=y, views=[COUNTS])

        cluster.add_chain(chains=chain_1)
        cluster.add_chain(chains=chain_2)
        self.assertItemsEqual(["ChainName1", "ChainName2"], cluster.keys())

        cluster.add_chain(chains=chain_3)
        cluster.add_chain(chains=chain_4)
        self.assertItemsEqual(["ChainName1", "ChainName2", "ChainName3", "ChainName4",],cluster.keys())

        #Check if the method incorrectly adds to the chain at the back of the list, along with / instead of saving over previous instance
        with self.assertRaises(IndexError):
            cluster.keys()[4]

    def test_add_multiple_chains_exceptions(self):
        y = self.minimum[0]
        x = self.minimum[1:3]

        cluster = Cluster(name="ClusterName")
        self.assertIsInstance(cluster, Cluster)
        self.assertItemsEqual([],cluster.keys())

        exception_message = "One or more of the supplied chains has an inappropriate type."

        chain_1 = self.stack0.get_chain(name="ChainName1", data_keys="Jan", x=x, y=y, views=[COUNTS])
        chain_2 = self.stack1.get_chain(name="ChainName2", data_keys="Feb", x=x, y=y, views=[COUNTS])
        chain_3 = self.stack2.get_chain(name="ChainName3", data_keys="Mar", x=x, y=y, views=[COUNTS])
        chain_4 = self.stack3.get_chain(name="ChainName4", data_keys="Apr", x=x, y=y, views=[COUNTS])

        self.assertItemsEqual([], cluster.keys())

        invalid_chains_list = [chain_1, chain_2, chain_3, chain_4, "This is not a chain"]
        with self.assertRaises(TypeError) as cm:
            cluster.add_chain(chains=invalid_chains_list)
        self.assertEquals(cm.exception.message, exception_message)

        # Assert that no chains were added
        self.assertItemsEqual([], cluster.keys())

    def test_add_multiple_chains(self):
        y = self.minimum[0]
        x = self.minimum[1:3]

        cluster = Cluster(name="ClusterName")
        self.assertIsInstance(cluster, Cluster)
        self.assertItemsEqual([],cluster.keys())

        chain_1 = self.stack0.get_chain(name="ChainName1", data_keys="Jan", x=x, y=y, views=[COUNTS])
        chain_2 = self.stack1.get_chain(name="ChainName2", data_keys="Feb", x=x, y=y, views=[COUNTS])
        chain_3 = self.stack2.get_chain(name="ChainName3", data_keys="Mar", x=x, y=y, views=[COUNTS])
        chain_4 = self.stack3.get_chain(name="ChainName4", data_keys="Apr", x=x, y=y, views=[COUNTS])

        cluster.add_chain(chains=[chain_1, chain_2, chain_3, chain_4])
        self.assertItemsEqual(["ChainName1", "ChainName2", "ChainName3", "ChainName4"], cluster.keys())

    def test_add_dataframe(self):

        df = self.example_data_A_data[self.one_of_each]

        cluster = Cluster('DataFrame')
        cluster.add_chain(df)

        #check that we have a cluster
        self.assertIsInstance(cluster, Cluster)
        self.assertItemsEqual('DataFrame', cluster.name)

        #check the cluster keys
        self.assertItemsEqual(['_'.join(self.one_of_each)], cluster.keys())

        #cluster contents
        for item in cluster.values():
            self.assertIsInstance(item, pd.DataFrame)

    # def test_add_multiple_dataframe(self):
    # to be added

    def test_dataframe_exceptions(self):
        y = self.minimum[0]
        x = self.minimum[1:3]
        chain_1 = self.stack0.get_chain(name="ChainName1", data_keys="Jan", x=x, y=y, views=[COUNTS])
        chain_2 = self.stack1.get_chain(name="ChainName2", data_keys="Feb", x=x, y=y, views=[COUNTS])
        chain_3 = self.stack2.get_chain(name="ChainName3", data_keys="Mar", x=x, y=y, views=[COUNTS])
        chain_4 = self.stack3.get_chain(name="ChainName4", data_keys="Apr", x=x, y=y, views=[COUNTS])
        df = self.example_data_A_data[self.one_of_each]

        exception_message = "One or more of the supplied chains has an inappropriate type."

        cluster = Cluster('chains_and_frames')
        invalid_chains_list = [chain_1, chain_2, chain_3, chain_4, df]        
        
        #check adding chains and dfs 
        with self.assertRaises(TypeError) as cm:
            cluster.add_chain(chains=invalid_chains_list)
        self.assertEquals(cm.exception.message, exception_message)

#     def test_save_cluster(self):
#         # TO DO -- Cluster saving probably not working yet
#         y = ['Gender']
#         x = ['Animal', 'Region']
#         cluster = Cluster(name="ClusterName")
# 
#         self.assertIsInstance(cluster, Cluster)
#         self.assertItemsEqual([],cluster.keys())
# 
#         for i in xrange(3):
#             chain = self.stack0.get_chain(name="ChainName{0}".format(i), data_keys="Jan", x=x, y=y, views=['default'])
#             cluster.add_chain(chains=chain)
# 
#         # Create a dictionary with the attribute structure of the cluster
#         cluster_attributes = test_helper.create_attribute_dict(cluster)
# 
#         cluster.save()
#         filename = "{0}.cluster".format(cluster.name)
#         loaded_cluster = Cluster.load(filename)
# 
#         # Create a dictionary with the attribute structure of the cluster
#         loaded_cluster_attributes = test_helper.create_attribute_dict(loaded_cluster)
# 
#         # Ensure that we are not comparing the same variable (in memory)
#         self.assertNotEqual(id(cluster), id(loaded_cluster))
# 
#         # Make sure that this is working by altering the loaded_stack_attributes
#         # and comparing the result. (It should fail)
# 
#         # Change a 'value' in the dict
#         loaded_cluster_attributes['__dict__']['name'] = "SomeOtherName"
#         with self.assertRaises(AssertionError):
#             self.assertEqual(cluster_attributes, loaded_cluster_attributes)
# 
#         # reset the value
#         loaded_cluster_attributes['__dict__']['name'] = cluster_attributes['__dict__']['name']
#         self.assertEqual(cluster_attributes, loaded_cluster_attributes)
# 
#         # Change a 'key' in the dict
#         del loaded_cluster_attributes['__dict__']['name']
#         loaded_cluster_attributes['__dict__']['new_name'] = cluster_attributes['__dict__']['name']
#         with self.assertRaises(AssertionError):
#             self.assertEqual(cluster_attributes, loaded_cluster_attributes)
# 
#         # reset the value
#         del loaded_cluster_attributes['__dict__']['new_name']
#         loaded_cluster_attributes['__dict__']['name'] = cluster_attributes['__dict__']['name']
#         self.assertEqual(cluster_attributes, loaded_cluster_attributes)
# 
#         # Remove a key/value pair
#         del loaded_cluster_attributes['__dict__']['name']
#         with self.assertRaises(AssertionError):
#             self.assertEqual(cluster_attributes, loaded_cluster_attributes)
# 
#         # Cleanup
#         if os.path.exists(filename):
#             os.remove(filename)
            
    def setup_stack_Example_Data_A(self, name=None, fk=None, xk=None, yk=None, views=None, weights=None):
        if name is None:
            name = "Example Data (A)"
        if fk is None:
            fk = ['no_filter']
        if xk is None:
            xk = self.minimum
        if yk is None:
            yk = ['@'] + self.minimum
        if views is None:
            views = ['cbase', 'counts']
        if not isinstance(weights, list):
            weights = [weights]
                
        stack = Stack(name=name)
        stack.add_data(
            data_key=stack.name, 
            meta=self.example_data_A_meta, 
            data=self.example_data_A_data
        )
        
        for weight in weights:
            stack.add_link(
                data_keys=stack.name,
                filters=fk,
                x=xk, 
                y=yk, 
                views=QuantipyViews(views),
                weights=weight
            )
        
        return stack    

    @classmethod
    def tearDownClass(self):
        filepath ='./tests/ClusterName.cluster'
        if os.path.exists(filepath):
            os.remove(filepath)
