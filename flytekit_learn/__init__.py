import logging
import re

from flytekit_learn.dataset import Dataset
from flytekit_learn.model import Model


# silence FlyteRemote beta warning since flytekit-learn won't expose it to end users.
class FlyteRemoteFilter(logging.Filter):
    def filter(self, record):
        return not re.match("^This feature is still in beta.+", record.getMessage())


# silence logger warnings having to do with PickleFile, since flytekit-learn allows this
# by default.
class PickleFilter(logging.Filter):
    def filter(self, record):
        return not re.match(".+Flyte will default to use PickleFile as the transport.+", record.getMessage())


flytekit_logger = logging.getLogger("flytekit")
flytekit_logger.addFilter(PickleFilter())

flytekit_remote_logger = flytekit_logger.getChild("remote")
flytekit_remote_logger.addFilter(FlyteRemoteFilter())
