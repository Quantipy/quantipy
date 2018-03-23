#!/usr/bin/python
# -*- coding: utf-8 -*-

OPTIONS = {
	'new_rules': False,
	'new_chains': False,
	'short_item_texts': False,
	'convert_chains': False,
	'fast_stack_filters': False
}

def set_option(option, val):
	"""
	"""
	if not option in OPTIONS:
		err = "'{}' is not a valid option!".format(option)
		raise ValueError(err)
	OPTIONS[option] = val
	return None

