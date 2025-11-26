#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("vecindex_finder/requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="vecindexfinder",
    version="0.1.0",
    author="VecIndexFinder Team",
    author_email="your.email@example.com",
    description="向量数据库索引参数推荐工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vecindexfinder",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "vecindex_finder": ["config.yml"],
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "vecindexfinder=index_finder:main",
        ],
    },
) 