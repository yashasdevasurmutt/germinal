from setuptools import setup, find_packages

"""
Minimal setup for local installation: pip install -e .
"""

setup(
    name="germinal",
    version="0.0.1",
    packages=find_packages(include=["germinal*"]),
    install_requires=[],
)
