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

    def add_chain(self, chains=None):
        """ Adds chains to a cluster """
        # If a single item was supplied, change it to a list of items
        if not isinstance(chains, (list, Chain, pd.DataFrame)):
            raise TypeError("You must pass either a Chain or a list of Chains to Cluster.add_chain()")
        else:

            if isinstance(chains, Chain):
                self[chains.name] = chains

            elif isinstance(chains, list) and all([isinstance(chain, Chain) for chain in chains]):
                # Ensure that all items in chains is of the type Chain.
                for chain in chains:
                    self[chain.name] = chain

            elif isinstance(chains, pd.DataFrame):
                    self['_'.join(chains.columns.tolist())] = chains

            else:
                # One or more of the items in chains is not a chain.
                raise TypeError("One or more of the supplied chains has an inappropriate type.")

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
