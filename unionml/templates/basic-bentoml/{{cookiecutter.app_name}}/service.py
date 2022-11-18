"""Create a BentoML API Service."""

from digits_classifier_app import service


service.load_model("latest", framework="sklearn")
svc = service.create(
    enable_async=True,
    supported_resources=("cpu", ),
    supports_cpu_multi_threading=False,
    runnable_method_kwargs={"batchable": False},
)
