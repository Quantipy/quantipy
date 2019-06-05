from .chain import Chain
import pickle
from collections import OrderedDict
import pandas as pd
import copy
from quantipy.core.tools.view.query import get_dataframe
from quantipy.core.helpers.functions import get_text
import os

class Cluster(OrderedDict):
    """
    Container class in form of an OrderedDict of Chains.

    It is possible to interact with individual Chains through the Cluster
    object. Clusters are mainly used to prepare aggregations for an export/
    build, e.g. MS Excel Workbooks. 
    """

    def __init__(self, name=""):
        super(Cluster, self).__init__()
        self.name = name

    def __setstate__(self, attr_dict):
        self.__dict__.update(attr_dict)

    def __reduce__(self):
        return self.__class__, (self.name, ), self.__dict__, None, iter(list(self.items()))

    def _verify_banked_chain_spec(self, spec):
        """
        Verify chain conforms to the expected banked chain structure.
        """
        if not type(spec) is dict:
            return False
        try:
            ctype = spec['type']
            cname = spec['name']
            ctext = spec['text']
            citems = spec['items']
            cbases = spec['bases']
            for c in citems:
                cichain = c['chain']
                citext = c['text']
        except:
            return False
        
        if not ctype=='banked-chain':
            return False
        if not isinstance(cname, str):
            return False        
        if not isinstance(ctext, dict):
            return False
        for key, value in list(ctext.items()):
            if not isinstance(key, str):
                return False
            if not isinstance(value, str):
                return False            
        if not isinstance(citems, list):
            return False
        if not isinstance(cbases, bool):
            return False            
        if not all([isinstance(item['chain'], Chain) for item in citems]):
            return False
        if not all([isinstance(item['text'], dict) for item in citems]):
            return False
        if not all([len(item['text'])>0 for item in citems]):
            return False
        for item in citems:
            for key, value in list(item['text'].items()):
                if not isinstance(key, str):
                    return False
                if not isinstance(value, str):
                    return False
                
        cview = spec.get('view', None)
        if cview is None:
            for c in citems:
                if 'view' in c:
                    if not isinstance(c['view'], str):
                        return False
                else:
                    return False
        else:
            if not isinstance(cview, str):
                return False

        return True

    def add_chain(self, chains=None):
        """ Adds chains to a cluster """
        # If a single item was supplied, change it to a list of items
        is_banked_spec = False
        if not isinstance(chains, (list, Chain, pd.DataFrame, dict)):
            raise TypeError(
                "You must pass either a Chain, a list of Chains or a"
                " banked chain definition (as a dict) into"
                " Cluster.add_chain().")
        elif isinstance(chains, dict):
            if chains.get('type', None)=='banked-chain':
                is_banked_spec = True
                if not self._verify_banked_chain_spec(chains):
                    raise TypeError(
                        "Your banked-chain definition is not correctly"
                        " formed. Please check it again.")
 
        if isinstance(chains, Chain):
            self[chains.name] = chains

        elif is_banked_spec:
            self[chains.get('name')] = chains

        elif isinstance(chains, list) and all([
            isinstance(chain, Chain) or \
            self._verify_banked_chain_spec(chain) 
            for chain in chains]):
            # Ensure that all items in chains is of the type Chain.
            for chain in chains:
                if chain.get('type', None)=='banked-chain':
                    self[chain.get('name')] = chain
                else:
                    self[chain.name] = chain

        elif isinstance(chains, pd.DataFrame):
            if any([
                    isinstance(idx, pd.MultiIndex) 
                    for idx in [chains.index, chains.columns]]):
                if isinstance(chains.index, pd.MultiIndex):
                    idxs = '_'.join(chains.index.levels[0].tolist())
                else:
                    idxs = chains.index
                if isinstance(chains.columns, pd.MultiIndex):
                    cols = '_'.join(chains.columns.levels[0].tolist())
                else:
                    idxs = chains.columns
                self['_|_'.join([idxs, cols])] = chains                
            else:
                self['_'.join(chains.columns.tolist())] = chains

        else:
            # One or more of the items in chains is not a chain.
            raise TypeError("One or more of the supplied chains has an inappropriate type.")

    def bank_chains(self, spec, text_key):
        """
        Return a banked chain as defined by spec.

        This method returns a banked or compound chain where the spec
        describes how the view results from multiple chains should be
        banked together into the same set of dataframes in a single
        chain.

        Parameters
        ----------
        spec : dict
            The banked chain specification object.  
        text_key : str, default='values'
            Paint the x-axis of the banked chain using the spec provided
            and this text_key.  

        Returns
        -------
        bchain : quantipy.Chain
            The banked chain.
        """
        
        if isinstance(text_key, str):
            text_key = {'x': [text_key]}
        
        chains = [c['chain'] for c in spec['items']]
        
        bchain = chains[0].copy()

        dk = bchain.data_key
        fk = bchain.filter
        xk = bchain.source_name
        yks = bchain.content_of_axis
        
        vk = spec.get('view', None)
        if vk is None:
            vk = spec['items'][0]['view']
        else:
            get_vk = False
            for i, item in enumerate(spec['items']):
                if not 'view' in item:
                    spec['items'][i].update({'view': vk})
        
        vks = list(set([item['view'] for item in spec['items']]))
        
        if len(vks)==1:
            notation = vks[0].split('|')
            notation[-1] = 'banked-{}'.format(spec['name'])
            bvk = '|'.join(notation)
        else:
            base_method = vks[0].split('|')[1]
            same_method = all([
                vk.split('|')[1]==base_method
                for vk in vks[1:]])
            if same_method:
                bvk = 'x|{}||||banked-{}'.format(base_method, spec['name'])                
            else:
                bvk = 'x|||||banked-{}'.format(spec['name'])
        
        for yk in list(bchain[dk][fk][xk].keys()):
            bchain[dk][fk][xk][yk][bvk] = bchain[dk][fk][xk][yk].pop(vks[0])
            bchain[dk][fk][xk][yk][bvk].name = bvk
        
        bchain.views = [
            vk_test
            for vk_test in bchain.views
            if 'cbase' in vk_test
        ]
        bchain.views.append(bvk)
        
        # Auto-painting approach
        idx_cbase = pd.MultiIndex.from_tuples([
            (get_text(spec['text'], text_key, 'x'), 'cbase')],
            names=['Question', 'Values'])
        
        # Non-auto-painting approach
#         idx_cbase = pd.MultiIndex.from_tuples([
#             (spec['name'], 'cbase')],
#             names=['Question', 'Values'])

        idx_banked = []
        banked = {}
        
        for yk in yks:
            banked[yk] = []
            for c, chain in enumerate(chains):
                xk = chain.source_name
                vk_temp = spec['items'][c]['view']
#                 print xk, yk, vk_temp
                df = get_dataframe(chain, keys=[dk, fk, xk, yk, vk_temp])
                if isinstance(idx_banked, list):
                    idx_banked.extend([
                        (spec['name'], '{}:{}'.format(xk, value[1])) 
                        for value in df.index.values
                    ])
                banked[yk].append(df)
            banked[yk] = pd.concat(banked[yk], axis=0)
            if banked[yk].columns.levels[1][0]=='@':
                banked[yk] = pd.DataFrame(
                    banked[yk].max(axis=1),
                    index=banked[yk].index,
                    columns=pd.MultiIndex.from_tuples(
                        [(spec['name'], '@')],
                        names=['Question', 'Values'])
                )
            
            xk = bchain.source_name
            if isinstance(idx_banked, list):
                banked_values_meta = [
                    {'value': idx[1], 'text': spec['items'][i]['text']} 
                    for i, idx in enumerate(idx_banked)]
                bchain.banked_meta = {
                    'name': spec['name'],
                    'type': spec['type'],
                    'text': spec['text'],
                    'values': banked_values_meta
                }
                # When switching to non-auto-painting, use this
#                 idx_banked = pd.MultiIndex.from_tuples(
#                     idx_banked, 
#                     names=['Question', 'Values'])
                # Auto-painting
                question_text = get_text(spec['text'], text_key, 'x')
                idx_banked = pd.MultiIndex.from_tuples([
                        (question_text, get_text(value['text'], text_key, 'x')) 
                        for i, value in enumerate(bchain.banked_meta['values'])], 
                    names=['Question', 'Values'])

            banked[yk].index = idx_banked
            bchain[dk][fk][xk][yk][bvk].dataframe = banked[yk]
            bchain[dk][fk][xk][yk][bvk]._notation = bvk
#             bchain[dk][fk][xk][yk][bvk].meta()['shape'] = banked[yk].shape
            bchain[dk][fk][xk][yk][bvk]._x['name'] = spec['name']
            bchain[dk][fk][xk][yk][bvk]._x['size'] = banked[yk].shape[0]
                
        bchain.name = 'banked-{}'.format(bchain.name)
        for yk in yks:
            for vk in list(bchain[dk][fk][xk][yk].keys()):
                if vk in bchain.views:                    
                    if 'cbase' in vk:
                        bchain[dk][fk][xk][yk][vk].dataframe.index = idx_cbase
                        bchain[dk][fk][xk][yk][vk]._x['name'] = spec['name']
                    
                else:
                    del bchain[dk][fk][xk][yk][vk]
        
        bchain[dk][fk][spec['name']] = bchain[dk][fk].pop(xk)
        
        bchain.props_tests = list()
        bchain.props_tests_levels = list()
        bchain.means_tests = list()
        bchain.means_tests_levels = list()
        bchain.has_props_tests = False
        bchain.has_means_tests = False
        bchain.annotations = None
        
        bchain.is_banked = True
        bchain.source_name = spec['name']
        bchain.banked_view_key = bvk
        bchain.banked_spec = spec
        for i, item in enumerate(spec['items']):
            bchain.banked_spec['items'][i]['chain'] = item['chain'].name
                
        return bchain

    def _build(self, type):
        """ The Build exports the chains using methods supplied with 'type'. """
        pass

    def merge(self):
        """
        Merges all Chains found in the Cluster into a new pandas.DataFrame.
        """
        orient = self[list(self.keys())[0]].orientation
        chainnames = list(self.keys())
        if orient == 'y':
            return pd.concat([self[chainname].concat()
                              for chainname in chainnames], axis=1)
        else:
            return pd.concat([self[chainname].concat()
                              for chainname in chainnames], axis=0)



    def save(self, path_cluster):
        """
        Load Stack instance from .stack file.

        Parameters
        ----------
        path_cluster : str
            The full path to the .cluster file that should be created, including
            the extension.

        Returns
        -------
        None
        """
        if not path_cluster.endswith('.cluster'):
            raise ValueError(
                "To avoid ambiguity, when using Cluster.save() you must provide the full path to "
                "the cluster file you want to create, including the file extension. For example: "
                "cluster.save(path_cluster='./output/MyCluster.cluster'). Your call looks like this: "
                "cluster.save(path_cluster='%s', ...)" % (path_cluster)
            )
        f = open(path_cluster, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    # STATIC METHODS

    @staticmethod
    def load(path_cluster):
        """
        Load Stack instance from .stack file.

        Parameters
        ----------
        path_cluster : str
            The full path to the .cluster file that should be created, including
            the extension.

        Returns
        -------
        None
        """
        if not path_cluster.endswith('.cluster'):
            raise ValueError(
                "To avoid ambiguity, when using Cluster.load() you must provide the full path to "
                "the cluster file you want to create, including the file extension. For example: "
                "cluster.load(path_cluster='./output/MyCluster.cluster'). Your call looks like this: "
                "cluster.load(path_cluster='%s', ...)" % (path_cluster)
            )
        f = open(path_cluster, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj