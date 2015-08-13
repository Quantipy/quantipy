from collections import defaultdict

class Cache(defaultdict):


    def __init__(self):
        # The 'lock_cache' raises an exception in the
        super(Cache, self).__init__(Cache)

    def __reduce__(self):
        return self.__class__, tuple(), None, None, self.iteritems()
