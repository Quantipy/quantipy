.. toctree::
 	:maxdepth: 5
	:includehidden:

=======================
Logic and set operaters
=======================

------
Ranges
------
The ``frange()`` function takes a string of abbreviated ranges, possibly delimited
by a comma (or some other character) and extrapolates its full,
unabbreviated list of ints.

>>> from quantipy.core.tools.dp.prep import frange

**Basic range**:

>>> frange('1-5')
[1, 2, 3, 4, 5]

**Range in reverse**:

>>> frange('15-11')
[15, 14, 13, 12, 11]

**Combination**:

>>> frange('1-5,7,9,15-11')
[1, 2, 3, 4, 5, 7, 9, 15, 14, 13, 12, 11]

**May include spaces for clarity**:

>>> frange('1-5, 7, 9, 15-11')
[1, 2, 3, 4, 5, 7, 9, 15, 14, 13, 12, 11]

-------------
Complex logic
-------------
Multiple conditions can be combined using ``union`` or ``intersection`` set
statements. Logical mappers can be arbitrarily nested as long as they are
well-formed.

``union``
---------
``union`` takes a list of logical conditions that will be treated with
**or** logic.

Where **any** of logic_A, logic_B **or** logic_C are ``True``:

>>> union([logic_A, logic_B, logic_C])

``intersection``
----------------
``intersection`` takes a list of conditions that will be
treated with  **and** logic.

Where **all** of logic_A, logic_B **and** logic_C are ``True``:

>>> intersection([logic_A, logic_B, logic_C])

"List" logic
------------
Instead of using the verbose ``has_any`` operator, we can express simple, non-nested
*or* logics simply as a list of codes. For example ``{"q1_1": [1, 2]}`` is an
example of list-logic, where ``[1, 2]`` will be interpreted as ``has_any([1, 2])``,
meaning if **q1_1** has any of the values **1** or **2**.

``q1_1`` has any of the responses 1, 2 or 3:

>>> l = {"q1_1": [1, 2, 3]}


``has_any``
-----------
``q1_1`` has any of the responses 1, 2 or 3:

>>> l = {"q1_1": has_any([1, 2, 3])}

``q1_1`` has any of the responses 1, 2 or 3 and no others:

>>> l = {"q1_1": has_any([1, 2, 3], exclusive=True)}


``not_any``
-----------
``q1_1`` doesn't have any of the responses 1, 2 or 3:

>>> l = {"q1_1": not_any([1, 2, 3])}

``q1_1`` doesn't have any of the responses 1, 2 or 3 but has some others:

>>> l = {"q1_1": not_any([1, 2, 3], exclusive=True)}

``has_all``
-----------
``q1_1`` has all of the responses 1, 2 and 3:

>>> l = {"q1_1": has_all([1, 2, 3])}

``q1_1`` has all of the responses 1, 2 and 3 and no others:

>>> l = {"q1_1": has_all([1, 2, 3], exclusive=True)}

``not_all``
-----------
``q1_1`` doesn't have all of the responses 1, 2 and 3:

>>> l = {"q1_1": not_all([1, 2, 3])}

``q1_1`` doesn't have all of the responses 1, 2 and 3 but has some others:

>>> l = {"q1_1": not_all([1, 2, 3], exclusive=True)}

``has_count``
-------------

``q1_1`` has exactly 2 responses:

>>> l = {"q1_1": has_count(2)}

``q1_1`` has 1, 2 or 3 responses:

>>> l = {"q1_1": has_count([1, 3])}

``q1_1`` has 1 or more responses:

>>> l = {"q1_1": has_count([is_ge(1)])}

``q1_1`` has 1, 2 or 3 responses from the response group 5, 6, 7, 8 or 9:

>>> l = {"q1_1": has_count([1, 3, [5, 6, 7, 8, 9]])}

``q1_1`` has 1 or more responses from the response group 5, 6, 7, 8 or 9:

>>> l = {"q1_1": has_count([is_ge(1), [5, 6, 7, 8, 9]])}

``not_count``
-------------
``q1_1`` doesn't have exactly 2 responses:

>>> l = {"q1_1": not_count(2)}

``q1_1`` doesn't have 1, 2 or 3 responses:

>>> l = {"q1_1": not_count([1, 3])}

``q1_1`` doesn't have 1 or more responses:

>>> l = {"q1_1": not_count([is_ge(1)])}

``q1_1`` doesn't have 1, 2 or 3 responses from the response group 5, 6, 7, 8 or 9:

>>> l = {"q1_1": not_count([1, 3, [5, 6, 7, 8, 9]])}

``q1_1`` doesn't have 1 or more responses from the response group 5, 6, 7, 8 or 9:

>>> l = {"q1_1": not_count([is_ge(1), [5, 6, 7, 8, 9]])}

----------------------------------
Boolean slicers and code existence
----------------------------------
``any()``, ``all()``
``code_count()``, ``is_nan()``
