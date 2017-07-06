.. toctree::
  :maxdepth: 5
  :includehidden:

================================
Inherited ``qp.DataSet`` methods
================================

As mentioned ``qp.Batch`` is a subclass of ``qp.DataSet`` and methods are
inherited. The important inherited methods are these which allow the manipulation
of the meta component. That means meta-edits can be done globally (run method 
on ``qp.DataSet``) or batchly (run method on ``qp.Batch``). Batch meta-edits 
always overwrite global meta-edits and while building a ``qp.Cluster`` for a 
``qp.Batch`` the modified meta information are taken from ``.meta_edits``.

The following methods can be used to create meta-edits for a ``qp.Batch``:

>>> batch.hiding('q2b', [2])
>>> batch.sorting('q2', fix=[97, 98])
>>> batch.slicing('q1', [1, 2, 3, 4, 5])
>>> batch.set_variable_text('gender', 'Gender???')
>>> batch.set_value_texts('gender', {1: 'Men', 2: 'Women'})
>>> batch.set_property('q1', 'base_text', 'This var has a second filter.')

Some methods, that can not be used to manipulate the meta component, are not
allowed to be used for a ``Batch``. These will raise, otherwise they can lead
to confusion and errors in the instance.
