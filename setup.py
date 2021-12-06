from setuptools import setup, find_packages

setup(
    name="flytekit-learn",
    version="0.0.0+dev0",
    description="The easiest way to build and deploy models.",
    author="unionai-oss",
    author_email="info@union.ai",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "flytekit>=0.24.0",
        "sklearn",
        "numpy",
        "pandas",
        "fastapi",
        "pydantic",
        "typer",
    ],
    entry_points = {
        "console_scripts": ["fklearn = flytekit_learn.cli:app"]
    },
)
