#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pdi",
    version="0.1",
    author='Lev Maximov',
    author_email='lev.maximov@gmail.com',
    url='https://github.com/axil/pdi',
    description="Helper functions from Pandas Illustrated article: find and findall",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['pandas'],
    packages=['pdi'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT License",
    zip_safe=False,
    keywords=['find', 'findall', 'series', 'pandas'],
)
