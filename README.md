# UnionML

The microframework for building and deploying machine learning services

## Build the Documentation

Until we release this into the wild, you'll have to build the docs locally.

First create and activate a virtual environment

```
python -m venv ~/unionml_env
source ~/unionml_env/bin/activate
```

Install requirements:

```
pip install -r requirements.txt -r requirements-dev.txt -r requirements-docs.txt
```

Make the documentation:

```
make -C docs clean html
```

Now you can open up `docs/_build/html/index.html` to view the docs.
