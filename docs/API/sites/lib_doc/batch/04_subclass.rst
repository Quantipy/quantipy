.. toctree::
  :maxdepth: 5
  :includehidden:

================================
Inherited ``qp.DataSet`` methods
================================

Being a ``qp.DataSet`` subclasss, ``qp.Batch`` inherits some of its methods.
The important ones are these which allow the manipulation of the meta component.
That means meta-edits can be applied globally (run methods on ``qp.DataSet``) or
``Batch``-specific (run methods on ``qp.Batch``). Batch meta-edits
always overwrite global meta-edits and while building a ``qp.Cluster`` from a
``qp.Batch``, the modified meta information is taken from ``.meta_edits``.

The following methods can be used to create meta-edits for a ``qp.Batch``:

>>> batch.hiding('q2b', [2])
>>> batch.sorting('q2', fix=[97, 98])
>>> batch.slicing('q1', [1, 2, 3, 4, 5])
>>> batch.set_variable_text('gender', 'Gender???')
>>> batch.set_value_texts('gender', {1: 'Men', 2: 'Women'})
>>> batch.set_property('q1', 'base_text', 'This var has a second filter.')

Some methods are not allowed to be used for a ``Batch``. These will raise a
``NotImplementedError`` to prevent inconsistent case and meta data states.
