from setuptools import find_packages, setup

__version__ = "0.0.0+develop"

setup(
    name="unionml",
    version=__version__,
    description="The easiest way to build and deploy machine learning services.",
    author="unionai-oss",
    author_email="info@union.ai",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "docker",
        "fastapi",
        "flytekit>=0.32.6",
        "gitpython",
        "joblib",
        "numpy",
        "pandas",
        "pydantic",
        "sklearn",
        "typer",
        "uvicorn",
    ],
    entry_points={"console_scripts": ["unionml = unionml.cli:app"]},
)
