import os
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info
from typing import Callable

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "fastapi")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

# ----- add metrics -----

instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

instrumentator.add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

instrumentator.add(
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

instrumentator.add(
    metrics.requests(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

# ----- custom metrics -----

def model_output(
    metric_name: str = "model_output",
    metric_doc: str = "Output value of model",
    metric_namespace: str = "",
    metric_subsystem: str = "",
    buckets=(100, 150, 200, 250, 300, 350, 400, float('inf')),
) -> Callable[[Info], None]:
    METRIC = Histogram(
        metric_name,
        metric_doc,
        labelnames=["model_type"],
        buckets=buckets,
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )
    METRIC.labels("LGBMRegressor")
    METRIC.labels("XGBRegressor")
    METRIC.labels("Neural_Network")

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/predict/{model_type}":
            predicted_eruption = info.response.headers.get("X-model-prediction")
            model_type=info.response.headers.get("X-model-type")
            if predicted_eruption:
                METRIC.labels(model_type).observe(float(predicted_eruption))

    return instrumentation

instrumentator.add(model_output(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
