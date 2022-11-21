(serving_bentoml)=

# Serving with BentoML

UnionML integrates with [BentoML](https://docs.bentoml.org/en/latest/index.html) to make the hand-off between
model training to production serving seamless.

````{admonition} Prerequisites
:class: important

First go through the following guides:

- {ref}`Local Training and Prediction <local_app>` guide for local model training.
- {ref}`Deploying to Flyte <flyte_cluster>` guide for model training at scale with [Flyte](https://flyte.org/).

Install the bentoml extra:

```{prompt} bash
:prompts: $

pip install unionml[bentoml]
```
````

## Setup

UnionML ships with a template that helps you get started with a bentoml-enabled unionml project:

```{prompt} bash
:prompts: $

unionml init basic_bentoml_app --template basic-bentoml
cd basic_bentoml_app
```

## Creating a BentoMLService

UnionML provides a {class}`~unionml.services.bentoml.BentoMLService` class that acts as a converter from
the components that you've defined in a UnionML app into a {class}`bentoml.Service`.

As you can see in our project template, we have a `digits_classifier_app.py` file that creates a UnionML app with a
{class}`~unionml.services.bentoml.BentoMLService`:

**`digits_classifier_app.py`**

```{literalinclude} ../../unionml/templates/basic-bentoml/{{cookiecutter.app_name}}/digits_classifier_app.py
---
lines: 1-33
emphasize-lines: 9,13
---
```

We can then train a model locally and save it to the local BentoML model store:

```{literalinclude} ../../unionml/templates/basic-bentoml/{{cookiecutter.app_name}}/digits_classifier_app.py
---
lines: 44-50
emphasize-lines: 6
---
```

If we run `python digits_classifier_app.py`, you should see output like this:

```{code-block}
LogisticRegression(max_iter=10000.0)
{'train': 1.0, 'test': 0.9722222222222222}
[6.0, 9.0, 3.0, 7.0, 2.0]
BentoML saved model: Model(tag="digits_classifier:degqqptj2g6jxlg6")
```

We've successfully saved our unionml-trained `model_object` to the BentoML model store under the tag
`digits_classifier:degqqptj2g6jxlg6`, where `digits_classifier` is the model name and `degqqptj2g6jxlg6` is the
version automatically created for us by BentoML.

```{note}
You can learn more about BentoML models and the model store {doc}`here <bentoml:concepts/model>`
```

### Defining a Model Service File

As a framework for creating and deploying ML-powered prediction services, BentoML enforces a clear boundary between
model training and serving.

UnionML adheres to this boundary by separating the UnionML app script and a BentoML service definition script. This is
so that we can flexibly iterate on model training and tuning, which is separate from serving the best model that we
trained.

In a separate file, we define which model we want to serve:

**`service.py`**

```{literalinclude} ../../unionml/templates/basic-bentoml/{{cookiecutter.app_name}}/service.py
```

Note that you can replace `"latest"` with an explicit model version, e.g. `"degqqptj2g6jxlg6"`, which may be a
desired practice if we want to deploy this service to production.

```{note}
Under the hood, the {meth}`~unionml.services.bentoml.BentoMLService.configure` method does the following:

- Creates a {class}`bentoml.Service` with a custom {class}`bentoml.Runnable` class that re-uses
  UnionML-defined components so that you can seamlessly create an API based on the
  {class}`unionml.dataset.Dataset.feature_loader`, {class}`unionml.dataset.Dataset.feature_transformer`,
  and {class}`unionml.model.Model.predictor` implementations.
- Infers the feature and output {doc}`API IO Descriptors <bentoml:reference/api_io_descriptors>`, based on the
  above UnionML-defined components. These can be explicitly provided as a keyword-argument to the `configure` method
  in case the feature and prediction output types are not recognized in the
  {attr}`~unionml.services.bentoml.BentoMLService.IO_DESCRIPTOR_MAPPING`.
- Defines a {attr}`~unionml.services.bentoml.BentoMLService.svc` property that can be used to access the underlying
  {class}`bentoml.Service`.
```

## Serving Locally

Start the server locally with:

```{prompt} bash
:prompts: $

bentoml serve service.py:service.svc
```

The UnionML `basic-bentoml` project template also comes with a `request.py` file that lets you test the local endpoint:

```{literalinclude} ../../unionml/templates/basic-bentoml/{{cookiecutter.app_name}}/request.py
```

Running it should hit the endpoint with a json payload that adheres to the BentoML Service API that we just defined:

```{prompt} bash
:prompts: $

python request.py
```

**Expected output:**

```{code-block}
[6.0,9.0,3.0,7.0,2.0]
```

```{note}
You can learn more about the `bentoml serve` command [here](https://docs.bentoml.org/en/latest/reference/cli.html#bentoml-serve)
```

## Building a Bento

A {doc}`Bento <bentoml:concepts/bento>` is a standardized file archive containing all the source code, models, data, and
additional artifacts that BentoML needs to deploy the model to some target infrastructure. To build a Bento, first we
need to define a `bentofile.yaml`:

```{literalinclude} ../../unionml/templates/basic-bentoml/{{cookiecutter.app_name}}/bentofile.yaml
```

```{note}
The `bentofile.yaml` file can be configured with additional options, which you can learn more about
[here](https://docs.bentoml.org/en/latest/concepts/bento.html#bento-build-options).
```

Then we simply invoke the [bentoml build](https://docs.bentoml.org/en/latest/reference/cli.html#bentoml-build) cli
command:

```{prompt} bash
:prompts: $

bentoml build
```

**Expected Output**

```{code-block}
Building BentoML service "digits_classifier:tdtkiddj22lszlg6" from build context "...".
Packing model "digits_classifier:degqqptj2g6jxlg6"

██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

Successfully built Bento(tag="digits_classifier:tdtkiddj22lszlg6").
```

Congratulations! You've now built a Bento, which is uniquely identified with the tag `digits_classifier:tdtkiddj22lszlg6`.
You can serve this Bento locally with the `bentoml serve` tag:

```{prompt} bash
:prompts: $

bentoml serve digits_classifier:tdtkiddj22lszlg6
```

## Deploying a Bento

BentoML offers three ways to deploy a Bento to production:

- 🐳 Containerize your Bento for custom docker deployment.
- 🦄 Yatai: A Kubernetes-native model deployment platform.
- 🚀 `bentoctl`: a command-line tool for deploying Bentos on any cloud platform.

To learn more about these deployment options, refer to the BentoML
[deployment guide](https://docs.bentoml.org/en/latest/concepts/deploy.html).
