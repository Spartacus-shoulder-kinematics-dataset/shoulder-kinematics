""" Setup.py file """

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="spartacus",
    version="0.0.1",
    author="Ipuch, ANaaim, FMoissenet",
    author_email="",
    description="Shoulder dataset treatment",
    long_description=long_description,
    python_requires=">=3.10",
    zip_safe=False,
    include_package_data=True,
    packages=find_packages(where="spartacus"),
    package_dir={"": "spartacus"},
    package_data={"": ["*.csv"]},
)
