import pathlib

from setuptools import find_packages, setup

__version__ = "0.0.0+dev0"


with open("requirements.txt") as f:
    install_requires = [x.strip() for x in f.readlines()]

LICENSE: str = "Apache"
README: str = pathlib.Path("README.md").read_text()

extras_require = {}
for extra in ["bentoml"]:
    with open(f"extras_require/{extra}.txt") as f:
        extras_require[extra] = [x.strip() for x in f.readlines()]

setup(
    name="unionml",
    version=__version__,
    author="unionai-oss",
    author_email="info@union.ai",
    description="The easiest way to build and deploy machine learning microservices.",
    long_description=README,
    long_description_content_type="text/markdown",
    license=LICENSE,
    keywords=["machine-learning", "artificial-intelligence", "microservices"],
    data_files=[("", ["LICENSE"])],
    include_package_data=True,
    packages=find_packages(
        include=["unionml*"],
        exclude=["tests*"],
    ),
    package_data={"unionml": ["py.typed"]},
    python_requires=">=3.7",
    platforms="any",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={"console_scripts": ["unionml = unionml.cli:app"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        f"License :: OSI Approved :: {LICENSE} Software License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    url="https://github.com/unionai-oss/unionml/",
    project_urls={
        "Documentation": "https://unionml.readthedocs.io/",
        "Source Code": "https://github.com/unionai-oss/unionml/",
        "Issue Tracker": "https://github.com/unionai-oss/unionml/issues",
    },
)
