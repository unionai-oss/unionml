(dataset)=

# Defining a Dataset

In {ref}`Initializing a UnionML App <initialize>`, we created a UnionML app project,
which contains an `app.py` script. In this guide, we'll learn how a `Dataset` is
defined and how we can customize its behavior.

## What's a UnionML Dataset?

A `Dataset` is one of the core parts of a *UnionML app*. You can think of
it as a specification for a dataset's source in addition to a set of common
machine-learning-specific abstractions, which we'll get into later in this guide.

First, let's define a dataset:

```{code-block} python
from unionml import Dataset

dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
```

```{note}
In the above code snippet you might notice a few things:

- We're defining a `Dataset` with the name `"digits_dataset"`.
- The `targets` argument accepts a list of strings referring to the column names.
  By default `unionml.Dataset` understands `pandas.DataFrame` objects as datasets,
  but as we'll see later this can be customized to accept data in any arbitrary format.
- The `test_size` argument specifies what fraction of the dataset should be reserved
  as the hold-out test set for model evaluation.
- The `shuffle` argument ensures that the data is shuffled before it's split into training
  and test sets, while `random_state` makes this shuffling process deterministic.
```

## Core `Dataset` Functions

In this toy example, we'll use the sklearn [digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
as our dataset.

```{important}
By default, the `Dataset` class understands how to work with `pandas.DataFrame` objects,
so in this section we'll assume that we're working with one. If you would like built-in support
for other data structures, please [create an issue](https://github.com/unionai-oss/unionml/issues/new)!
```

### `reader`

When working with `pandas.DataFrame`s, the only `Dataset` method you need to implement is
the `reader`, which specifies how to get your training data. This is done by decorating a
function with the `dataset.reader` decorator.

```{code-block} python
import pandas as pd
from sklearn.datasets import load_digits

@dataset.reader
def reader(sample_frac: float = 1.0, random_state: int = 12345) -> pd.DataFrame:
    data = load_digits(as_frame=True).frame
    return data.sample(frac=sample_frac, random_state=random_state)
```

Notice how we can define any arbitrary set of arguments. In this case, we can choose to
sample the digits dataset to produce a subset of data.

### `loader`

The `loader` function should specify how to load the output of the `reader` function into memory.
Since the `Dataset` class knows how to handle `pandas.DataFrame`s, defining the `loader` function
is optional if you're working with them.

However, suppose that we refactor our reader function so that it returns a parquet file. This
is where the [Flyte Type System](https://docs.flyte.org/projects/cookbook/en/latest/auto/core/type_system/flyte_python_types.html#sphx-glr-auto-core-type-system-flyte-python-types-py) comes in handy.
We can use [FlyteFile](https://docs.flyte.org/projects/flytekit/en/latest/generated/flytekit.types.file.FlyteFile.html#flytekit.types.file.FlyteFile) as the output annotation of the `reader` like so:

```{code-block} python
import pandas as pd
from flytekit.types.file import FlyteFile
from sklearn.datasets import load_digits

@dataset.reader
def reader(sample_frac: float = 1.0, random_state: int = 12345) -> FlyteFile:
    data = load_digits(as_frame=True).frame
    output_path = "./digits.parquet"
    data.to_parquet(output_path)
    return FlyteFile(path=output_path)
```

Then to read the file back into memory, we specify our loader:

```{code-block} python
@dataset.loader
def loader(data: FlyteFile) -> pd.DataFrame
    with open(data) as f:
        return pd.from_parquet(f)
```

```{admonition} Why do we need two separate steps?
:class: important

When using UnionML-supported data structures (such as `pandas.DataFrame`s and other supported
[Flyte Types](https://docs.flyte.org/projects/cookbook/en/latest/auto/core/type_system/flyte_python_types.html#sphx-glr-auto-core-type-system-flyte-python-types-py)), it automatically
understands how to handle the serialization/deserialization across the data reading and
model training functions.

For unrecognized types, UnionML will use [Pickle Type](https://docs.flyte.org/projects/cookbook/en/latest/auto/core/type_system/flyte_pickle.html#sphx-glr-auto-core-type-system-flyte-pickle-py) as the
fallback, which is not guaranteed to work across different versions of Python or different
versions of your package dependencies.

With that context, there are two cases where you want to define a `reader` and `loader` function:

1. When the most natural way of storing a dataset is in files or a directory structure.
2. When you don't want to use pickle as the data transfer format between data reading and model training.
```

### `splitter`

The `splitter` function should specify how to split your data into train and test sets. When
working with `pandas.DataFrame`s you can supply the `test_size`, `shuffle`, and `random_state`
arguments to the `Dataset` initializer to split your data as a fraction of `test_size`.

If `shuffle == True` then the dataframe is shuffled before splitting using `random_state` as
the random seed.

To implement your own splitting behavior, you can use the `dataset.splitter` decorator. The
following example is roughly equivalent to the built-in behavior:

```{code-block} python
from typing import NamedTuple

Splits = NamedTuple("Splits", train=pd.DataFrame, test=pd.DataFrame)

@dataset.splitter
def splitter(data: pd.DataFrame, test_size: float, shuffle: bool, random_state: int) -> Splits:
    if shuffle:
        data = data.sample(frac=1.0, random_state=random_state)
    n = int(data.shape[0] * test_size)
    return data.iloc[:-n], data.iloc[-n:]
```

```{note}
The splitter is expected to return an indexable type whose underlying type matches the
output of `reader`. In this case, we return a `NamedTuple` of `pd.DataFrame`s.
```

### `parser`

Finally, this specifies how to extract features and targets from your dataset.
By supplying the `features` and `targets` arguments for the `Dataset` initializer,
you're indicating which columns in the `pandas.DataFrame` are features and which
are targets, respectively.

```{note}
If you only supply the `targets` argument, the `Dataset` assumes that the rest
of the columns in the dataframe are features.
```

Similar to the `dataset.splitter` decorator, you can use the `dataset.parser` decorator
to implement your own parser. The following example is roughly equivalent to the built-in
behavior:

```{code-block} python
from typing import Optional

Parsed = NamedTuple("Parsed", features=pd.DataFrame, targets=pd.DataFrame)

@dataset.parser
def parser(data: pd.DataFrame, features: Optional[List[str]], targets: List[str]) -> Parsed:
    if not features:
        features = [col for col in data if col not in targets]
    return data[features], data[targets]
```

## `Dataset` Functions for Prediction

The following functions define behavior for prediction across multiple use cases.

### `feature_loader`

Similar to the `loader` function, the `feature_loader` function handles the loading data into memory
from a file or from some raw data format.

The default feature loader is equivalent to the following:

```{code-block} python
import json
from typing import Any, Dict, List, Union
from pathlib import Path

RawFeatures = List[Dict[str, Any]]

@dataset.feature_loader
def feature_loader(features: Union[Path, RawFeatures]) -> pd.DataFrame:
    if isinstance(features, Path):
        # handle case where `features` input is a filepath
        with features.open() as f:
            features: RawFeatures = json.load(f)
    return pd.DataFrame(features)
```

Note that this function handles the case where the input is a file path or a list
of dictionary records.

### `feature_transformer`

The `feature_transformer` function handles additional processing steps performed on the
output of `feature_loader` in case you want to do some stateless transforms, like normalizing
the values of your feature data based on static parameters.

For example, suppose we received an image in the form of a dataframe, where pixel values are
in the range 0 to 256. To normalize the data to be between 0 and 1, we'd specify a function like this:

```{code-block} python
@dataset.feature_transformer
def feature_transformer(data: pd.DataFrame) -> pd.DataFrame:
    return data / 255
```


## Next

Now that we've defined a `Dataset`, we need to {ref}`Bind a Model and Dataset <model>` together
to create our UnionML app.
