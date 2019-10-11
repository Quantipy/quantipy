#!/usr/bin/python
# -*- coding: utf-8 -*-


def set_encoding(encoding):
    """
    Hack sys.setdefaultencoding() to escape ASCII hell.

    Parameters
    ----------
    encoding : str
        The name of the encoding to default to.
    """
    default_stdout = sys.stdout
    default_stderr = sys.stderr
    reload(sys)
    sys.setdefaultencoding(encoding)
    sys.stdout = default_stdout
    sys.stderr = default_stderr


def make_like_ascii(text):
    """
    Replaces any non-ascii unicode with ascii unicode.

    http://www.fileformat.info/info/unicode/char/
    """
    unicode_ascii_mapper = {
        u'\u2022': u'-',
        u'\u2013': u'-',
        u'\u2018': u'\u0027',
        u'\u2019': u'\u0027',
        u'\u201c': u'\u0022',
        u'\u201d': u'\u0022',
        u'\u00a3': u'GBP',
        u'\u20AC': u'EUR',
        u'\u2026': u'\u002E\u002E\u002E',
    }
    for old, new in unicode_ascii_mapper.iteritems():
        text = text.replace(old, new)
    return text


# -----------------------------------------------------------------------------
# i/o
# -----------------------------------------------------------------------------
def cpickle_copy(obj):
    copy = pickle.loads(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
    return copy


def load_json(path_json, hook=OrderedDict):
    with open(path_json) as f:
        obj = unicoder(json.load(f, object_pairs_hook=hook))
        return obj


def loads_json(json_text, hook=OrderedDict):
    obj = json.loads(json_text, object_pairs_hook=hook)
    return obj


def save_json(obj, path_json, decode_str=False, decoder='UTF-8'):
    if decode_str:
        obj = unicoder(obj, decoder)

    def represent(obj):
        if isinstance(obj, np.generic):
            return np.asscalar(obj)
        else:
            return "Unserializable object: {}".format(type(obj))
    with open(path_json, 'w+') as f:
        json.dump(obj, f, default=represent, sort_keys=True)


def load_csv(path_csv):
    data = pd.DataFrame.from_csv(path_csv)
    return data


# -----------------------------------------------------------------------------
# lists
# -----------------------------------------------------------------------------
def frange(range_def, sep=','):
    """
    Return the full, unabbreviated list of ints suggested by range_def.

    This function takes a string of abbreviated ranges, possibly
    delimited by a comma (or some other character) and extrapolates
    its full, unabbreviated list of ints.

    Parameters
    ----------
    range_def : str
        The range string to be listed in full.
    sep : str, default=','
        The character that should be used to delimit discrete entries in
        range_def.

    Returns
    -------
    res : list
        The exploded list of ints indicated by range_def.
    """
    res = []
    for item in range_def.split(sep):
        if '-' in item:
            a, b = item.split('-')
            a, b = int(a), int(b)
            lo = min([a, b])
            hi = max([a, b])
            ints = range(lo, hi+1)
            if b <= a:
                ints = list(reversed(ints))
            res.extend(ints)
        else:
            res.append(int(item))
    return res


def flatten_list(the_list):
    flat = []
    for item in the_list:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def ensure_list(obj):
    if not obj:
        obj = []
    elif not isinstance(obj, list):
        obj = [obj]
    return obj


def uniquify_list(the_list):
    seen = []
    new_list = []
    for item in the_list:
        if item not in seen:
            new_list.append(item)
        else:
            seen.append(item)
    return new_list
