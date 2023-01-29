#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pandas-illustrated',
    version='0.3',
    author='Lev Maximov',
    author_email='lev.maximov@gmail.com',
    url='https://github.com/axil/pandas-illustrated',
    description='Helper functions from the Pandas Illustrated guide',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.7",
    install_requires=[
        'pandas',
        'bs4',              # for patch_series
        'lxml',             # for bs4 
        'IPython',          # for sidebyside
    ],
    packages=['pdi'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',    
        'Programming Language :: Python :: 3.10',    
        'Programming Language :: Python :: 3.11',    
    ],
    license='MIT License',
    zip_safe=False,
    keywords=['find', 'findall', 'insert', 'drop', 'levels', 'sidebyside', 'pandas'],
)
