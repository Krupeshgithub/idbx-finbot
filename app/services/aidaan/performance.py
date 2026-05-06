import contextvars
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Standard logger for module tracking (stdout)
logger = logging.getLogger(__name__)

# Dedicated performance logger (performance.log)
perf_logger = logging.getLogger("performance")

@dataclass
class PerformanceMetric:
    """
    Enhanced performance metric with request binding.
    """
    component: str
    details: str = ""
    tokens: str = "N/A"
    request_id: str = ""  # NEW: Bind to originating request
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    latency: float = 0.0

    def finish(self):
        self.end_time = time.monotonic()
        self.latency = self.end_time - self.start_time

# Context variable to hold metrics grouped by request_id
request_metrics: contextvars.ContextVar[Dict[str, List[PerformanceMetric]]] = contextvars.ContextVar(
    "request_metrics", 
    default={}
)

def start_metric(component: str, details: str = "", tokens: str = "N/A", request_id: str = "") -> PerformanceMetric:
    """
    Start tracking a new performance metric.
    """
    metric = PerformanceMetric(component, details, tokens, request_id)
    metrics_dict = request_metrics.get().copy()
    
    if request_id not in metrics_dict:
        metrics_dict[request_id] = []
    
    metrics_dict[request_id].append(metric)
    request_metrics.set(metrics_dict)
    return metric

def record_metric(component: str, latency: float, details: str = "", tokens: str = "N/A", request_id: str = ""):
    """
    Record a completed performance metric.
    """
    metric = PerformanceMetric(component, details, tokens, request_id)
    metric.latency = latency
    
    metrics_dict = request_metrics.get().copy()
    
    if request_id not in metrics_dict:
        metrics_dict[request_id] = []
    
    metrics_dict[request_id].append(metric)
    request_metrics.set(metrics_dict)

def get_metrics(request_id: Optional[str] = None) -> List[PerformanceMetric]:
    """
    Get metrics for a specific request or all metrics.
    """
    metrics_dict = request_metrics.get()
    
    if request_id:
        return metrics_dict.get(request_id, [])
    
    # Return all metrics flattened
    all_metrics = []
    for metrics_list in metrics_dict.values():
        all_metrics.extend(metrics_list)
    return all_metrics

def clear_metrics(request_id: Optional[str] = None):
    """
    Clear metrics for a specific request or all metrics.
    """
    if request_id:
        metrics_dict = request_metrics.get().copy()
        metrics_dict.pop(request_id, None)
        request_metrics.set(metrics_dict)
    else:
        request_metrics.set({})

def format_metrics_table(request_id: Optional[str] = None) -> str:
    """
    Format metrics as a markdown table.
    """
    metrics = get_metrics(request_id)
    if not metrics:
        return "No performance metrics recorded."

    header = "| Service / Component | Tokens Used | Time Taken (Seconds) | Details & Actions Performed |"
    separator = "| :--- | :--- | :--- | :--- |"
    rows = []
    total_latency = 0.0
    
    # Estimate total tokens by summing numeric values
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

def get_performance_summary(request_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate performance summary for monitoring dashboard.
    """
    metrics = get_metrics(request_id)
    
    if not metrics:
        return {
            "total_latency": 0.0,
            "total_tokens": 0,
            "component_breakdown": {},
            "llm_turns": 0,
            "tool_calls": 0,
        }
    
    total_tokens = 0
    component_breakdown = {}
    
    for m in metrics:
        # Aggregate by component
        if m.component not in component_breakdown:
            component_breakdown[m.component] = {
                "latency": 0.0,
                "tokens": 0,
                "count": 0,
            }
        
        component_breakdown[m.component]["latency"] += m.latency
        component_breakdown[m.component]["count"] += 1
        
        # Parse tokens
        try:
            if str(m.tokens).isdigit():
                token_count = int(m.tokens)
                component_breakdown[m.component]["tokens"] += token_count
                total_tokens += token_count
            elif str(m.tokens).startswith("~") and str(m.tokens)[1:].isdigit():
                token_count = int(m.tokens[1:])
                component_breakdown[m.component]["tokens"] += token_count
                total_tokens += token_count
        except:
            pass
    
    return {
        "request_id": request_id or "all",
        "total_latency": sum(m.latency for m in metrics),
        "total_tokens": total_tokens,
        "component_breakdown": component_breakdown,
        "llm_turns": len([m for m in metrics if "LLM" in m.component]),
        "tool_calls": len([m for m in metrics if "Alpha Vantage" in m.component or "FinBERT" in m.component]),
        "metric_count": len(metrics),
    }

def log_performance(summary: str, request_id: Optional[str] = None):
    """
    Unified entry point for performance logging.
    Logs to stdout (via standard logger) and to performance.log (via perf_logger).
    """
    prefix = f"[{request_id}] " if request_id else ""
    
    # 1. Standard log (for stdout/app.log)
    logger.info(f"*************\n{prefix}{summary}\n*************")
    
    # 2. Performance log (for performance.log)
    perf_logger.info(f"{prefix}{summary}")
