import logging
import re

from unionml.dataset import Dataset
from unionml.model import Model

# necessary for single sourcing version
try:
    from importlib import metadata  # type: ignore
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata  # type: ignore

# single source version from setup.py
__title__ = __name__
__version__ = metadata.version(__title__)  # type: ignore


# silence FlyteRemote beta warning since unionml won't expose it to end users.
class FlyteRemoteFilter(logging.Filter):
    def filter(self, record):
        return not re.match("^This feature is still in beta.+", record.msg)


# silence logger warnings having to do with PickleFile, since unionml allows this
# by default.
class PickleFilter(logging.Filter):
    def filter(self, record):
        return not re.match(".+Flyte will default to use PickleFile as the transport.+", record.msg)


flytekit_logger = logging.getLogger("flytekit")
flytekit_logger.addFilter(PickleFilter())

flytekit_remote_logger = flytekit_logger.getChild("remote")
flytekit_remote_logger.addFilter(FlyteRemoteFilter())
