#!/usr/bin/env python
from setuptools import find_namespace_packages, setup

setup(
    packages=find_namespace_packages(
        include=["llama_iris", "llama_iris.*"]
    ),
    install_requires=[
        "llama_index>=0.9.40",
        "sqlalchemy-iris>=0.12.0",
    ],
    python_requires=">3.8,<3.12",
)
