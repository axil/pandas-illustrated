#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pandas-illustrated',
    version='0.1',
    author='Lev Maximov',
    author_email='lev.maximov@gmail.com',
    url='https://github.com/axil/pandas-illustrated',
    description='Helper functions from the Pandas Illustrated Guide',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'pandas',
        'bs4',              # for patch_series
        'lxml',             # for bs4 
        'IPython',          # for sidebyside
    ],
    packages=['pdi'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT License',
    zip_safe=False,
    keywords=['find', 'findall', 'insert', 'drop', 'sidebyside', 'pandas'],
)
