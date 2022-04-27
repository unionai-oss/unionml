(dataset)=

# Defining a Dataset

A `Dataset` is one of the core parts of a *`unionml` app*. You can think of
it as a specification for a dataset's source in addition to a set of common
machine-learning-specific abstractions, which we'll get into later in this guide.

First, let's define a dataset:

```{code-block} python
from unionml import Dataset

dataset = Dataset(
    name="digits_dataset",
    test_size=0.2,
    shuffle=True,
    random_state=42,
    targets=["target"],
)
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

## `Dataset` Functions

By default, the `Dataset` object understands how to work with `pandas.DataFrame` objects,
so in this section we'll assume that we're working with one.

In this toy example, we'll use the sklearn [digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
as our dataset.

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

## Next

Now that we've defined a `Dataset`, we need to {ref}`Bind a Model and Dataset <model>` together
to create our `unionml` app.
