#-*- coding: utf-8 -*-

"""
Created on 21 Nov 2017
"""

import json
import pandas as pd
import numpy as np
import quantipy as qp

qp_dim_languages = {
	'en-GB': 'ENG',
	'sv-SE': 'SVE',
	'da-DK': 'DAN',
	'fi-FI': 'FIN',
	'nb-NO': 'NOR',
	'de-DE': 'DEU',
	'fr-FR': 'FRA',
	'zh-CN': 'CHS',
	'id-ID': 'IND',
	'ms-MY': 'MSL',
	'th-TH': 'THA'
}


class DimLabels():
	"""
	"""

	def __init__(self, name, text_key='en-GB'):

		self.name = name
		self.text_key = text_key
		self.text = {}
		self.labels = []
		self.incl_languages = []
		self.incl_labeltypes = []

	def add_text(self, text_object, replace=True):
		if isinstance(text_object, str):
			text_object = {self.text_key: text_object}
		self.text = text_object
		self.labels_from_text(replace)
		self._lang_ltype_from_label(replace)
		return None

	def _lang_ltype_from_label(self, replace=True):
		if replace:
			self.incl_languages = []
			self.incl_labeltypes = []
		for lab in self.labels:
			if not lab.language in self.incl_languages:
				self.incl_languages.append(lab.language)
			if lab.labeltype and not lab.labeltype in self.incl_labeltypes:
				self.incl_labeltypes.append(lab.labeltype)
		return None

	def labels_from_text(self, replace=True):
		if replace: self.labels = []
		for item in list(self.text.items()):
			if isinstance(item[1], dict):
				for e_item in list(item[1].items()):
					dimlabel = DimLabel(e_item, item[0], self.text_key)
					if not self._label_exists(dimlabel):
						self.labels.append(dimlabel)
			else:
				dimlabel = DimLabel(item, None, self.text_key)
				if not self._label_exists(dimlabel):
					self.labels.append(dimlabel)
		return None

	def _label_exists(self, label):
		return any(d_l.language == label.language and
		           d_l.labeltype == label.labeltype
		           for d_l in self.labels)


class DimLabel():
	"""
	"""

	def __init__(self, text=None, edit=None, text_key=None):

		self.text = ''
		self.language = ''
		self.default_lan = qp_dim_languages.get(text_key, 'ENG')
		self.labeltype = None
		if text:
			self.to_dim(text, edit)

	def to_dim(self, text, edit=None):
		if isinstance(text, str):
			self.language = self.default_lan
			self.text = text
		else:
			self.language = qp_dim_languages.get(text[0], 'ENG')
			self.text = text[1]
		self.text = self.text.replace('\n', ' ').replace('"', '')
		self.labeltype = edit
		return None












