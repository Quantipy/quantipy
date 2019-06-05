from collections import defaultdict

class Cache(defaultdict):


    def __init__(self):
        # The 'lock_cache' raises an exception in the
        super(Cache, self).__init__(Cache)

    def __reduce__(self):
        return self.__class__, tuple(), None, None, iter(list(self.items()))


    def set_obj(self, collection, key, obj):
    	'''
    	Save a Quantipy resource inside the cache.

    	Parameters
    	----------
    	collection : {'matrices', 'weight_vectors', 'quantities',
    				  'mean_view_names', 'count_view_names'}
    		The key of the collection the object should be placed in.
    	key : str
    		The reference key for the object.
    	obj : Specific Quantipy or arbitrary Python object.
    		The object to store inside the cache.

    	Returns
    	-------
    	None
    	'''
    	self[collection][key] = obj

    def get_obj(self, collection, key):
    	'''
    	Look up if an object exists in the cache and return it.

    	Parameters
    	----------
    	collection : {'matrices', 'weight_vectors', 'quantities',
    				  'mean_view_names', 'count_view_names'}
    		The key of the collection to look into.
    	key : str
    		The reference key for the object.

    	Returns
    	-------
    	obj : Specific Quantipy or arbitrary Python object.
    		The cached object mapped to the passed key.
    	'''
    	if collection == 'matrices':
    		return self[collection].get(key, (None, None))
    	elif collection == 'squeezed':
    		return self[collection].get(key, (None, None, None, None, None, None, None))
    	else:
    		return self[collection].get(key, None)
