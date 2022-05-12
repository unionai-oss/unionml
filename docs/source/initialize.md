(initialize)=

# Initializing a UnionML App

UnionML ships with app templates that you can use to quickly set up a UnionML app.
In this guide, you'll learn a little bit more about the anatomy of a complete UnionML app
project.

## Basic App Template

Let's create a simple app that performs hand-written digits classification.
UnionML ships with a command-line interface that you can use to quickly create an app:

```{code-block} bash
unionml init my_app
```

You should see a new directory `my_app` with the following structure:

```{code-block} bash
my_app
├── Dockerfile          # docker image for packaging up your app for deployment
├── app.py              # app script
└── requirements.txt    # python dependencies for your app
```

## Create a Virtual Environment

Create a virtual environment for your app so that you can isolate its dependencies:

```{code-block} bash
cd my_app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To make sure everything's working as expected, we can run `app.py` as a Python script:

```{code-block} bash
python my_app.py
```

````{dropdown} Expected output

   ```{code-block}
      LogisticRegression(max_iter=100.0)
      {'train': 1.0, 'test': 0.9666666666666667}
      [6.0, 9.0, 3.0, 7.0, 2.0]
   ```

````

## Next

Now that we've created our UnionML app, we can now go deeper into how it works by looking
at how a {ref}`Dataset object is defined <dataset>`.
