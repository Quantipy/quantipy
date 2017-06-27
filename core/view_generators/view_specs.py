from quantipy.core.tools.qp_decorators import modify
from collections import OrderedDict
from operator import add, sub, mul, div

@modify(to_list='text_key')
def net(append_to=[], condition=None, text='', text_key=None):
    """
    Add a well-formed instruction dict for net to a net_map.

    Parameters
    ----------
    append_to: list or dict (list item)
        If a list is provided, the defined net is appended. If a list item is
        provided, the new text is added to the existing net. 
    condition: list / dict (complex logic)
        List codes to band categorical answers in a net group. Use complex 
        logic if external variables are involved.
    text: str or dict
        Text for the net. If a str is provided, a text_key is required.
        In a dict more than one text_key can be specified, for example:
        text = {'en-GB': 'the first net', 'de-DE': 'das erste net'}
    text_key: str, list of str
        If text is a str, it will be added for all defined text_keys.    
    """
    if not (isinstance(text, dict) or text_key): 
        raise ValueError("'text' must be a dict or a text_key must be provided.")
    elif not isinstance(text, dict):
        text = {tk: text for tk in text_key}
    if isinstance(append_to, dict):
        append_to['text'].update(text)
    else:
        net = {len(append_to)+1: condition, 'text': text}
        append_to.append(net)
        return append_to

def calc(expression, text, text_key=None, exclusive=False):
    """
    Produce a well-formed instruction dict for a calculated net.

    At least two net-like groups get connected via a mathematical operator 
    ('+', '-', '*', '/').

    Parameters
    ----------
    expression: tuple 
        At least two net-like groups get connected via a mathematical operator 
        ('+', '-', '*', '/'). The groups are called by position, for example:
        expression = (3, '-', 1)
    text: str or dict
        Text for the calculated net. If a str is provided, a text_key is required.
        In a dict more than one text_key can be specified, for example:
        text = {'en-GB': 'NPS', 'de-DE': 'Net Promotor Score'}
    text_key: str, list of str
        If text is a str, it will be added for all defined text_keys.  
    exclusive: bool, default False
        If True the groups are suppressed and only the calculation result is kept.
    """
    if not (isinstance(text, dict) or text_key): 
        raise ValueError("'text' must be a dict or a text_key must be provided.")
    elif not isinstance(text, dict):
        text = {tk: text for tk in text_key}
    operator = {'+': add, '-': sub, '*': mul, '/': div} 
    instruction = OrderedDict([('calc', tuple(operator.get(e, 'net_{}'.format(e)) 
                                                           for e in expression)),
                   ('calc_only', exclusive),
                   ('text', text)])
    return instruction
