from chain import Chain
import cPickle
from collections import OrderedDict
import pandas as pd

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

    def _verify_banked_chain_definition(chain):
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

        if not isinstance(cviews, list):
            return False

        cview = chain.get('view', None)
        if view is None:
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
        if isinstance(chains, dict):
            if not self._verify_banked_chain_definition(chains):
                raise TypeError(
                    "Your banked-chain definition is not correctly"
                    " formed. Please check it again.")
        else:

            if isinstance(chains, Chain):
                self[chains.name] = chains

            elif isinstance(chains, list) and all([
                isinstance(chain, Chain) or \
                self.verify_banked_chain_definition(chain) 
                for chain in chains]):
                # Ensure that all items in chains is of the type Chain.
                for chain in chains:
                    self[chain.name] = chain

            elif isinstance(chains, pd.DataFrame):
                    self['_'.join(chains.columns.tolist())] = chains

            else:
                # One or more of the items in chains is not a chain.
                raise TypeError("One or more of the supplied chains has an inappropriate type.")

    def bank_chains(self, spec):

        dk = chains[0].data_key
        fk = chains[0].filter
        yks = chains[0].content_of_axis
        
        if not isinstance(vks, (list, tuple)):
            vks = [vks]
        
        banked = {}
        for yk in yks:
            banked[yk] = []
            for chain in chains:
                xk = chain.source_name
                banked[yk].append(
                    pd.concat([
                        get_dataframe(chain, keys=[dk, fk, xk, yk, vk])
                        for vk in vks],
                        axis=0)
                    )
            banked[yk] = pd.concat(banked[yk], axis=0)
            if banked[yk].columns.levels[1][0]=='@':
                banked[yk] = pd.DataFrame(
                    banked[yk].max(axis=1),
                    index=banked[yk].index,
                    columns=pd.MultiIndex.from_tuples(
                        [(chains[0].source_name, '@')],
                        names=['Question', 'Values'])
                )
        
        return banked        

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