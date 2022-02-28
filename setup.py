from setuptools import find_packages, setup

setup(
    name="flytekit-learn",
    version="0.0.0+dev0",
    description="The easiest way to build and deploy machine learning services.",
    author="unionai-oss",
    author_email="info@union.ai",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "flytekit>=0.30.3",
        "sklearn",
        "numpy",
        "pandas",
        "fastapi",
        "pydantic",
        "typer",
    ],
    entry_points={"console_scripts": ["fklearn = flytekit_learn.cli:app"]},
)
