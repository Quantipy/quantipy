from chain import Chain
import cPickle
from collections import OrderedDict
import pandas as pd
import copy
from quantipy.core.tools.view.query import get_dataframe
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
        return self.__class__, (self.name, ), self.__dict__, None, self.iteritems()

    def _verify_banked_chain_definition(self, chain):
        """
        Verify chain conforms to the expected banked chain structure.
        """
        if not isinstance(chain, dict):
            return False
        try:
            ctype = chain['type']
            cname = chain['name']
            ctext = chain['text']
            citems = chain['items']
            for c in citems:
                cichain = c['chain']
                citext = c['text']
        except:
            return False

        if not all([isinstance(c, Chain) for c in citems['chain']]):
            return False

        if not all([isinstance(t, dict) for t in citems['text']]):
            return False

        cview = chain.get('view', None)
        if cview is None:
            for c in citems:
                if not 'view' in c:
                    return False
        else:
            if not isinstance(cview, str):
                return False

        return True

    def add_chain(self, chains=None):
        """ Adds chains to a cluster """
        # If a single item was supplied, change it to a list of items
        if not isinstance(chains, (list, Chain, pd.DataFrame, dict)):
            raise TypeError(
                "You must pass either a Chain, a list of Chains or a"
                " banked chain definition (as a dict) into"
                " Cluster.add_chain().")
        elif isinstance(chains, dict):
            if chains.get('type', None)=='banked-chain':
                if not self._verify_banked_chain_definition(chains):
                    raise TypeError(
                        "Your banked-chain definition is not correctly"
                        " formed. Please check it again.")
 
        if isinstance(chains, Chain):
            self[chains.name] = chains

        elif isinstance(chains, list) and all([
            isinstance(chain, Chain) or \
            self._verify_banked_chain_definition(chain) 
            for chain in chains]):
            # Ensure that all items in chains is of the type Chain.
            for chain in chains:
                self[chain.name] = chain

        elif isinstance(chains, pd.DataFrame):
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
        
        chains = [c['chain'] for c in spec['items']]
        
        path_temp_chain = './banked.chain'
        chains[0].save(path_temp_chain)
        bchain = Chain().load(path_temp_chain)
        os.remove(path_temp_chain)

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
            bvk = 'x|||||banked-{}'.format(spec['name'])
        
        for yk in bchain[dk][fk][xk].keys():
            bchain[dk][fk][xk][yk][bvk] = bchain[dk][fk][xk][yk].pop(vks[0])
            bchain[dk][fk][xk][yk][bvk].name = bvk
        
        bchain.views = [
            vk_test
            for vk_test in bchain.views
            if 'cbase' in vk_test
        ]
        bchain.views.append(bvk)
        
        idx_cbase = pd.MultiIndex.from_tuples([
                (spec['text'][text_key], 'cbase')
                for vk_cbase in bchain.views
                if 'cbase' in vk_cbase
            ],
            names=['Question', 'Values'])

        banked = {}
        for yk in yks:
            banked[yk] = []
            for c, chain in enumerate(chains):
                xk = chain.source_name
                vk_temp = spec['items'][c]['view']
#                 print xk, yk, vk_temp
                banked[yk].append(
                    get_dataframe(chain, keys=[dk, fk, xk, yk, vk_temp]))
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
            banked[yk].index = [
                (spec['text'][text_key], item['text'][text_key])
                for item in spec['items']
            ]
            bchain[dk][fk][xk][yk][bvk].dataframe = banked[yk]
            bchain[dk][fk][xk][yk][bvk].meta()['shape'] = banked[yk].shape
        
        bchain.name = 'banked-{}'.format(bchain.name)
        for yk in yks:
            for vk in bchain[dk][fk][xk][yk].keys():
                if not vk in bchain.views:
                    del bchain[dk][fk][xk][yk][vk]
                if 'cbase' in vk:
                    bchain[dk][fk][xk][yk][vk].dataframe.index = idx_cbase
                    
        bchain.is_banked = True
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
        orient = self[self.keys()[0]].orientation
        chainnames = self.keys()
        if orient == 'y':
            return pd.concat([self[chainname].concat()
                              for chainname in chainnames], axis=1)
        else:
            return pd.concat([self[chainname].concat()
                              for chainname in chainnames], axis=0)



    def save(self, path="./"):
        """ Save's the Cluster object. """
        f = open(path+self.name+'.cluster', 'wb')
        cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
        f.close()

    # STATIC METHODS

    @staticmethod
    def load(filename):
        """
        Load a pickled Cluster instance.

        Attributes:
            filename ( string )
              Specifies the name of the file to be loaded.
              Example of use: loaded_cluster = Cluster.load(filepath)
        """
        f = open(filename, 'rb')
        obj = cPickle.load(f)
        f.close()
        return obj