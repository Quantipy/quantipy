from setuptools import setup, find_packages


setup(
    name='quantipy',
    version='0.1.1',
    # author='',
    # author_email='',
    packages=find_packages(exclude=['tests', 'quantipy.sandbox']),
    include_package_data=True,
    # tests_require=[
    #     'pytest',
    #     # 'pytest-cov',
    #     # 'pytest-xdist',
    # ],
    # setup_requires=[
    #     'pytest-runner',
    # ],
    install_requires=[
        'pandas>=0.19.2',
        # 'scipy', 
        'ftfy==4.4.3',
        'xmltodict',
        'lxml',
        'xlsxwriter',
        'pillow',
        'prettytable',
        'decorator',
        'watchdog',
        'requests',
        'python-pptx',
    ],
)

