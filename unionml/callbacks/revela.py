import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union
from urllib.parse import urlparse

import pandas as pd
from requests import Session
from typing_extensions import TypedDict

from unionml._logging import logger as unionml_logger

logger = unionml_logger.getChild("callbacks.revela")
logger.setLevel(os.environ.get("UNIONML_LOGLEVEL_REVELA", "WARNING"))


class RevelaTabularData(TypedDict):
    data: List[Sequence[Union[float, int, bool, str, None]]]
    columns: List[str]
    index: Sequence[Union[float, int, str]]


class RevelaDataPayload(TypedDict):
    deployment_id: str
    input_data: RevelaTabularData
    predictions: RevelaTabularData


def _default_serialize_list(data: List[Any]) -> RevelaTabularData:
    if not data:
        return {"data": [], "columns": [], "index": []}

    if all(isinstance(row, dict) for row in data):

        # Collect and sort all columns first
        keys: Set = set()
        row: Dict
        for row in data:
            keys = keys.union(row.keys())
        columns = sorted(keys)

        return {
            "index": list(range(len(data))),
            "columns": columns,
            # Convert to a list of lists, with None for missing values.
            "data": [[row.get(c, None) for c in columns] for row in data],
        }

    elif all(isinstance(row, (list, tuple)) for row in data):
        # Number columns based on the max length of any row.
        n_cols = len(max(data, key=len))
        columns = [str(col) for col in range(n_cols)]
        index = list(range(len(data)))

        return {
            "index": index,
            "columns": columns,
            # Slice + [None] prevents IndexError when row is shorter than columns.
            # and [None]*-N is a no-op.
            "data": [row[0:n_cols] + [None] * (len(row) - n_cols) for row in data],
        }

    elif all(isinstance(row, (int, float, bool, str, type(None))) for row in data):
        return {
            "index": list(range(len(data))),
            "columns": ["0"],
            # If all rows are primitives, we'll just return
            # a single (potentially mixed-type) column.
            "data": [[x] for x in data],
        }

    row_types = set(type(row) for row in data)
    raise NotImplementedError(f"serialization of list containing types: {row_types}.")


def _default_serialize_series(data: pd.Series) -> RevelaTabularData:
    return data.to_frame().convert_dtypes().to_dict(orient="split")


def _default_serialize_dataframe(data: pd.DataFrame) -> RevelaTabularData:
    return data.convert_dtypes().to_dict(orient="split")


@dataclass
class RevelaLogger:
    """A UnionML callback that logs data to Revela.

    This class can be used to log runtime predictions to Revela's ML
    monitoring API. Instances of this class can directly provided as a callback on
    any Model.predictor function. You must provide a deployment ID to use when
    logging data.

    Follow the [Revela API documentation](https://revela.app/docs) for more
    information about using the Revela API.

    Args:
        `deployment_id`: The ID of the Revela deployment to log data to.
        `api_token`: The API token to use when authenticating with the Revela API.
            Defaults to the value of the `REVELA_API_TOKEN` environment variable.
        `api_url`: The URL of the Revela API. Defaults to https://revela.app/api/next.
        `serializers`: A mapping of types to functions that can serialize those types
            into a `RevelaTabularData` object. Default serializations are provided for:
                - pandas.DataFrame
                - pandas.Series
                - list of rows of data comprised of dicts, lists, or primitives.
            . You can provide custom serializers for
            your own types here using this argument. All custom serializers must
            accept a single argument and return a `RevelaTabularData` object.
    """

    deployment_id: str
    api_token: str = field(default_factory=lambda: os.environ["REVELA_API_TOKEN"])
    api_url: str = "https://revela.app/api/next"
    serializers: Dict[type, Callable[[Any], RevelaTabularData]] = field(
        default_factory=lambda: {
            pd.DataFrame: _default_serialize_dataframe,
            pd.Series: _default_serialize_series,
            list: _default_serialize_list,
        }
    )

    def __post_init__(self):
        # If the user provided custom serializers, but didn't include the
        # default ones, add them back to the serializer map.
        self.serializers.update(
            {
                pd.DataFrame: self.serializers.get(
                    pd.DataFrame,
                    _default_serialize_dataframe,
                ),
                pd.Series: self.serializers.get(
                    pd.Series,
                    _default_serialize_series,
                ),
                list: self.serializers.get(
                    list,
                    _default_serialize_list,
                ),
            }
        )

        # Normalize the API URL.
        url = urlparse(self.api_url.rstrip("/"))
        if not url.path or url.path == "/":
            url = url._replace(path="/api/next")
        self.api_url = url.geturl()

        # Create a session to use for all requests.
        self.session = Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_token}"})

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"initialized for endpoint: {self.api_url}")

    def _serialize(self, data) -> Optional[RevelaTabularData]:
        try:
            return self.serializers.get(type(data), lambda x: None)(data)
        except Exception as err:
            logger.warning(
                " ".join(
                    [
                        f"Could not serialize object of type <{type(data)}>.",
                        "You may need to add a custom serializer.",
                        f"{type(err)}: {err}",
                    ]
                )
            )
            return None

    def __call__(
        self,
        model_object: Any,
        features: Any,
        predictions: Any,
    ):
        if logger.isEnabledFor(logging.DEBUG):
            params = {
                "model_object": type(model_object),
                "features": type(features),
                "predictions": type(predictions),
            }
            signature = ", ".join(f"{k}: {v}" for k, v in params.items())
            logger.debug(f"Invoked with: <Signature ({signature})>")

        input_data = self._serialize(features)
        predictions = self._serialize(predictions)

        if input_data is None:
            logger.warning(
                "Feature serialization returned nothing. Nothing available to send to Revela.",
            )
            return
        if predictions is None:
            logger.warning(
                "Prediction serialization returned nothing. Data will not be sent to Revela.",
            )
            return

        resp = self.session.post(
            f"{self.api_url}/data",
            json={
                "deployment_id": self.deployment_id,
                "input_data": input_data,
                "predictions": predictions,
            },
        )

        if not resp.ok:
            logger.warning(f"Response {resp.status_code}: {resp.text}")
        else:
            logger.info(f"Logged data to Revela deployment {self.deployment_id}.")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Logged payload: {resp.request.body}.")
