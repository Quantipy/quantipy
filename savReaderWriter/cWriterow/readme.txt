# cWriterow is a faster C version of the Python pyWriterow method.
# It requires that Cython + a C compiler be installed (http://docs.cython.org/src/quickstart/install.html)		
easy_install cython  # requires setuptools
python setup.py build_ext --inplace
