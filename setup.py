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
    package_data={
        "templates": ["*"],
    },
    install_requires=install_requires,
    entry_points={"console_scripts": ["unionml = unionml.cli:app"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache License",
        "Intended Audience :: Engineering/Science",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Machine Learning/Artificial Intelligence",
    ],
)
