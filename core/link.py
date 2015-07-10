#-*- coding: utf-8 -*-
import numpy as np
from view_generators.view_maps import QuantipyViews as View


class Link(dict):
    """
    The Link object is a subclassed dictionary that generates an instance of
    Pandas.DataFrame for every view method applied
    """
    def __init__(self,
                 the_filter,
                 y,
                 x,
                 data_key,
                 stack,
                 views=None,
                 store_view=False,
                 create_views=True):

        self.filter = the_filter
        self.y = y
        self.x = x
        self.data_key = data_key
        self.stack = stack

        # If this variable is set to true, then the view will be transposed.
        self.transpose = False

        if isinstance(views, str):
            views = View(views)
        elif isinstance(views, list):
            views = View(*views)
        elif views is None:
            views = View()

        if store_view:
            self.view = views

        data = stack[data_key].data
        if create_views:
            if '@1' not in data.keys():
                data['@1'] = np.ones(len(data.index))
            views._apply_to(self, weights)

    def get_meta(self):
        stack = self.stack
        data_key = self.data_key
        return stack[data_key].meta

    def get_data(self):
        stack = self.stack
        data_key = self.data_key
        filter_def = self.filter
        return stack[data_key][filter_def].data

    def __getitem__(self, key):
        """ The 'get' method for the Link(dict)

            If the 'transpose' variable is set to True THEN this method tries
            to transpose the result.

            Note: Only the numpy.T method has been implemented.
        """
        val = dict.__getitem__(self, key)

        if self.transpose:
            if "T" in dir(val):
                return val.T
            else:
                return val
        else:
            return val
