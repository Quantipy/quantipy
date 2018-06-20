# encoding: utf-8

from pptx.util import (
    Emu,
    Pt,
    Cm,
    Inches)

import pptx_defaults as pptx
import re

def update_dict_from_dict(update_dict, from_dict):
    """
    Updates keys in dict with values from other dict
    :param
        update_dict:
        from_dict:
    :return: None, update_dict are updated inplace
    """
    for key, value in from_dict.iteritems():
        if isinstance(value, dict):
            update_dict_from_dict(update_dict[key], from_dict[key])
        else:
            update_dict[key] = from_dict[key]


class PptxDefaults(object):
    """
    Handles all defaults for Class PptxPainter
    """

    def __init__(self):
        self._shapes = pptx.shapes
        self._charts = pptx.shapes['charts']
        self._textboxes = pptx.shapes['textboxes']
        self._chart_bar = pptx.shapes['charts']['bar']
        self._chart_bar_stacked100 = pptx.shapes['charts']['bar_stacked100']
        self._chart_line = pptx.shapes['charts']['line']
        self._chart_column = pptx.shapes['charts']['column']
        self._chart_pie = pptx.shapes['charts']['pie']
        self._textbox_header = pptx.shapes['textboxes']['header']
        self._textbox_footer = pptx.shapes['textboxes']['footer']

    def update_shape(self, shape, settings):
        """
        Updates self._[shape] with settings from matching dict

        Parameters
        ----------
        shape : str
            A string with the shape to update

        settings : dict
            A dictionary matching one or more keys in self._[shape]

        Returns
        -------
        None
            self._[shape] is updated
        """
        parameter_map = {'shapes': self._shapes,
                      'charts': self._charts,
                      'textboxes': self._textboxes,
                      'chart_bar': self._chart_bar,
                      'chart_bar_stacked100': self._chart_bar_stacked100,
                      'chart_line': self._chart_line,
                      'chart_column': self._chart_column,
                      'chart_pie': self._chart_pie,
                      'textbox_header': self._textbox_header,
                      'textbox_footer': self._textbox_footer,
                      }

        available_shapes = parameter_map.keys()
        shape = re.sub(' +', '', shape)
        if shape not in available_shapes:
            error_text = "Shape: {} is not an available shape. \n Available shapes are {}"
            raise ValueError(error_text.format(shape, available_shapes))

        update_dict_from_dict(parameter_map[shape], settings)