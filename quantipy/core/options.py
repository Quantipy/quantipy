#!/usr/bin/python
# -*- coding: utf-8 -*-

from .tools.logger import get_logger
logger = get_logger(__name__)

OPTIONS = {
	"modules_old": False
}

def set_option(option, val):
    """
    """
    if option not in OPTIONS:
        err = "'{}' is not a valid option".format(option)
        logger.error(err); ValueError(err)
    OPTIONS[option] = val
    return None