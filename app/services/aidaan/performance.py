import contextvars
import time
from typing import Dict, List, Any, Optional

class PerformanceMetric:
    def __init__(self, component: str, details: str = "", tokens: str = "N/A"):
        self.component = component
        self.tokens = tokens
        self.details = details
        self.start_time = time.monotonic()
        self.end_time: Optional[float] = None
        self.latency: float = 0.0

    def finish(self):
        self.end_time = time.monotonic()
        self.latency = self.end_time - self.start_time

# Context variable to hold the list of metrics for the current request
request_metrics: contextvars.ContextVar[List[PerformanceMetric]] = contextvars.ContextVar("request_metrics", default=[])

def start_metric(component: str, details: str = "", tokens: str = "N/A") -> PerformanceMetric:
    metric = PerformanceMetric(component, details, tokens)
    metrics = request_metrics.get().copy()
    metrics.append(metric)
    request_metrics.set(metrics)
    return metric

def record_metric(component: str, latency: float, details: str = "", tokens: str = "N/A"):
    metric = PerformanceMetric(component, details, tokens)
    metric.latency = latency
    metrics = request_metrics.get().copy()
    metrics.append(metric)
    request_metrics.set(metrics)

def get_metrics() -> List[PerformanceMetric]:
    return request_metrics.get()

def clear_metrics():
    request_metrics.set([])

def format_metrics_table() -> str:
    metrics = get_metrics()
    if not metrics:
        return "No performance metrics recorded."

    header = "| Service / Component | Tokens Used | Time Taken (Seconds) | Details & Actions Performed |"
    separator = "| :--- | :--- | :--- | :--- |"
    rows = []
    total_latency = 0.0
    
    # We estimate total tokens by summing if they are numeric
    total_tokens = 0
    
    for m in metrics:
        rows.append(f"| {m.component} | {m.tokens} | {m.latency:.3f}s | {m.details} |")
        total_latency += m.latency
        try:
            if str(m.tokens).isdigit():
                total_tokens += int(m.tokens)
            elif str(m.tokens).startswith("~") and str(m.tokens)[1:].isdigit():
                total_tokens += int(m.tokens[1:])
        except:
            pass

    rows.append(f"| **Total Pipeline Latency** | **~{total_tokens}** | **~{total_latency:.3f}s** | **End-to-End processing time.** |")
    
    return "\n".join([header, separator] + rows)
