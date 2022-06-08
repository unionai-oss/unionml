(basics)=

# Basics

```{toctree}
---
hidden: true
---
initialize
dataset
model
local_app
```

This section of the documentation introduces the main concepts of UnionML and aims to show you
the core pieces that together make up a UnionML app.

## UnionML Apps

A *UnionML app* is composed of two objects: {class}`~unionml.dataset.Dataset` and
{class}`~unionml.model.Model`. Together, they expose method decorator entrypoints that serve as the core
building blocks of an end-to-end machine learning application. The following sections will show you how to:

- {ref}`Initialize a UnionML App <initialize>`: quickly create a UnionML app with
  <a href="cli_reference.html#unionml-init">unionml init</a>.
- {ref}`Define a Dataset <dataset>`: specify where to *read* data from and how to *split* it into training and test
  sets, *parse* out features and targets, and *iterate* through batches of data.
- {ref}`Bind a Model and Dataset <model>`: specify how to *initialize*, *train*, *evaluate*, and generate
  *predictions* from a model given a {class}`~unionml.dataset.Dataset`.
- {ref}`Train and Predict Locally <local_app>`: perform training and prediction using a UnionML as a regular
  python script, then serve a `FastAPI` app for prediction.
