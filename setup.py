from setuptools import find_packages, setup

__version__ = "0.0.0+dev0"

with open("requirements.txt") as f:
    install_requires = [x.strip() for x in f.readlines()]

setup(
    name="unionml",
    version=__version__,
    description="The easiest way to build and deploy machine learning services.",
    author="unionai-oss",
    author_email="info@union.ai",
    packages=find_packages(exclude=["tests*"]),
    install_requires=install_requires,
    entry_points={"console_scripts": ["unionml = unionml.cli:app"]},
)
