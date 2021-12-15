"""Remote entrypoint for flytekit-learn."""

from flytekit.remote import FlyteRemote

def remote(obj):
    return RemoteModel(obj)


class RemoteModel():
    
    def __init__(self, model, config_file_path=None, project=None, domain=None):
        self.model = model

    def train(self):
        pass

    def predict(self):
        pass

    def serve(self):
        pass

    def remote(self, config_file_path = None, project = None, domain = None):
        self._remote = FlyteRemote.from_config(
            config_file_path=config_file_path,
            default_project=project,
            default_domain=domain,
        )