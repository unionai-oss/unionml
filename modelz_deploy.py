import typing
from dataclasses import dataclass

import requests


@dataclass
class DockerConfig:
    image: str


@dataclass
class GitConfig:
    branch: str
    builder: str
    directory: str
    image: str
    image_tag: str
    repo: str
    revision: str


@dataclass
class HuggingfaceConfig:
    space: str
    tag: str


@dataclass
class DeploymentSourceConfig:
    docker: DockerConfig
    git: GitConfig
    huggingface: HuggingfaceConfig


@dataclass
class DeploymentConfig:
    command: str
    deployment_source: DeploymentSourceConfig
    env_vars: typing.Optional[dict] = {}
    framework: str = "mosec"
    http_probe_path: typing.Optional[str] = None
    id: typing.Optional[str] = None
    max_replicas: int = 1
    min_replicas: int = 0
    name: str
    port: int = 0
    secret: typing.Optional[str] = None
    server_resource: str = "nvidia-tesla-t4-4c-16g"
    spot_instance: bool = False
    startup_duration: int = 0
    target_load: int = 0
    templateId: typing.Optional[str] = None
    zero_duration: int = 0


def deploy_model(name: str, image: str):
    """Deploy model to Modelz

    https://modelz-api.readme.io/reference/post_users-login-name-deployments-1
    """
    url = "https://union-ai.modelz.dev/api/v1/users/login_name/deployments"
    api_key = "mzi-921b803eb4931c569153b46dc8356a1a"

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "X-API-KEY": api_key,
    }
    payload = {
        "spec": {
            # "command": "string",
            "deployment_source": {
                "docker": {"image": image},
                # "git": {
                #     "branch": "string",
                #     "builder": "string",
                #     "directory": "string",
                #     "image": "string",
                #     "image_tag": "string",
                #     "repo": "string",
                #     "revision": "string"
                # },
                # "huggingface": {
                #     "space": "string",
                #     "tag": "string"
                # }
            },
            # "env_vars": { "additionalProp": "string" },
            "framework": "mosec",
            # "http_probe_path": "string",
            # "id": "string",
            "max_replicas": 1,
            "min_replicas": 0,
            "name": name,
            # "port": 0,
            # "secret": ["string"],
            "server_resource": "nvidia-tesla-t4-4c-16g",
            # "spot_instance": True,
            # "startup_duration": 0,
            # "target_load": 0,
            "templateId": "string",
            "zero_duration": 300,
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    print(response.text)


if __name__ == "__main__":
    deploy_model(name="modelz-test0-deployment-sd", image="ghcr.io/cosmicbboy/modelz-stable-diffusion:v1")
