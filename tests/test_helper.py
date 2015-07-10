from collections import defaultdict

def create_attribute_dict(obj):
    """ Takes a dict object and returns a ordereddict with the attributes
        from the object stored in the ['__dict__'] location

        Example:
            >> ...
            >> attr = create_attribute_dict(stack)
            >> attr.keys()
               ['__dict__', 'Jan']
            >> attr['__dict__']
               {'name':'The Stack Name',
                'stack_pos':'stack_root',
                'encoding':'UTF-8'
                   ...       ...
               }

    """
    attributes = defaultdict(dict)
    if isinstance(obj, dict):
        for key in obj.keys():
            attr_dict = defaultdict(dict)
            if hasattr(obj[key], '__dict__'):
                for attr in obj.__dict__:
                    if attr not in ['parent', 'view', 'data', 'stack', '_OrderedDict__root', '_OrderedDict__map']:
                        attr_dict[attr] = obj.__dict__[attr]
                attributes['__dict__'] = attr_dict
            attributes[key] = create_attribute_dict(obj[key])
    return attributes
