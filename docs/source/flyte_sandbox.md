(flyte_sandbox)=

# Spinning Up a Flyte Sandbox

In {ref}`Local Training and Prediction <local_app>` we ran our training and prediction
functions locally by:

1. Executing our `flytekit-learn` app as python module by calling the `model.train` and
   `model.predict` methods.
2. Starting a FastAPI server and invoking the `/train` and `/predict.` endpoints using
   the `requests` library.

```{note}
In the previous guides, we used our individual workstation to do the heavy lifting of
model training and prediction. While this might work well for prototyping, small datasets,
and light-weight models, this won't scale well to larger datasets and models. For that,
we'll need to access cloud resources.
```

`flytekit-learn` integrates tightly with [Flyte](https://docs.flyte.org/en/latest/), which is
a data- and machine-learning-aware orchestration platform that leverages cloud services like
[AWS](https://aws.amazon.com/) and [GCP](https://cloud.google.com/) to easily scale and
maintain data processing machine learning workloads.

In this guide, we'll:

1. Spin up a Flyte sandbox, which is a standalone, minimal Flyte cluster that you can create
   on your individual laptop or workstation.
2. Configure digit classification `flytekit-learn` app to use the Flyte sandbox as the compute
   backend for our training and prediction workload.


## Initialize a Flyte Sandbox

````{tabs}

   ```{tab} Brew - OSX

      Install [`flytectl`](https://docs.flyte.org/projects/flytectl/en/latest/index.html), the
      commandline interface for Flyte.

      ```{prompt} bash
      :prompts: $

      brew install flyteorg/homebrew-tap/flytectl
      ```

      Start the server, assuming the `flytekit-learn` app is in a `main.py` script

      ```{code-block} bash
      uvicorn main:app --reload
      ```

      Invoke the endpoints using the `requests` library

      ```{literalinclude} ../../tests/integration/sklearn/api_requests.py
      ```

   ```

   ```{tab} Curl

      Bind a FastAPI `app` to the `model` object with `model.serve`

      ```{literalinclude} ../../tests/integration/sklearn/fastapi_app.py
      ---
      lines: 3-6
      ---
      ```

      Start the server, assuming the `flytekit-learn` app is in a `main.py` script

      ```{code-block} bash
      uvicorn main:app --reload
      ```

      Invoke the endpoints using the `requests` library

      ```{literalinclude} ../../tests/integration/sklearn/api_requests.py
      ```

   ```

````
