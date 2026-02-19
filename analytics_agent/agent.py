"""
analytics_agent/agent.py

Analytics Agent — handles metrics, trends, growth reports, and dashboards.
A2A server on port 8891 using ADK + to_a2a.
"""

import json
import os
from dotenv import load_dotenv
import uvicorn

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from shared.utils import build_oci_model

load_dotenv()

oci_model = build_oci_model()

# ── Analytics tools ───────────────────────────────────────────────────────────

def calculate_growth_rate(current: float, previous: float) -> dict:
    """
    Calculate period-over-period growth rate.

    Args:
        current: Current period value.
        previous: Previous period value.

    Returns:
        dict with growth_percent and trend direction.
    """
    if previous == 0:
        return {"status": "error", "error_message": "Previous value cannot be zero."}
    growth = ((current - previous) / abs(previous)) * 100
    trend = "📈 Growing" if growth > 0 else "📉 Declining" if growth < 0 else "➡️ Flat"
    return {
        "status": "success",
        "growth_percent": round(growth, 2),
        "trend": trend,
        "current": current,
        "previous": previous,
    }


def generate_metrics_report(data_json: str) -> dict:
    """
    Generate a structured metrics report from JSON data.

    Args:
        data_json: A JSON string with metric names as keys and numeric values.
                   Example: '{"revenue": 50000, "users": 1200, "churn": 5.2}'

    Returns:
        dict with summary stats and top/bottom performers.
    """
    try:
        data = json.loads(data_json)
        if not isinstance(data, dict):
            return {"status": "error", "error_message": "Expected a JSON object."}

        numeric = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
        if not numeric:
            return {"status": "error", "error_message": "No numeric metrics found."}

        top = max(numeric, key=numeric.get)
        bottom = min(numeric, key=numeric.get)
        avg = sum(numeric.values()) / len(numeric)

        return {
            "status": "success",
            "total_metrics": len(numeric),
            "average_value": round(avg, 2),
            "top_metric": f"{top} = {numeric[top]}",
            "bottom_metric": f"{bottom} = {numeric[bottom]}",
            "all_metrics": numeric,
        }
    except json.JSONDecodeError:
        return {"status": "error", "error_message": "Invalid JSON string provided."}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def analyze_trend(values_csv: str) -> dict:
    """
    Analyze a time-series trend from comma-separated values.

    Args:
        values_csv: Comma-separated numeric values in chronological order.
                    Example: "100, 120, 115, 140, 160"

    Returns:
        dict with trend direction, average, min, max, and overall change.
    """
    try:
        values = [float(v.strip()) for v in values_csv.split(",") if v.strip()]
        if len(values) < 2:
            return {"status": "error", "error_message": "Need at least 2 data points."}

        overall_change = ((values[-1] - values[0]) / abs(values[0])) * 100
        avg = sum(values) / len(values)
        trend = "📈 Upward" if overall_change > 2 else "📉 Downward" if overall_change < -2 else "➡️ Sideways"

        return {
            "status": "success",
            "data_points": len(values),
            "trend": trend,
            "overall_change_percent": round(overall_change, 2),
            "average": round(avg, 2),
            "min": min(values),
            "max": max(values),
            "first": values[0],
            "last": values[-1],
        }
    except ValueError:
        return {"status": "error", "error_message": "All values must be numeric."}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def build_dashboard_summary(title: str, metrics_json: str) -> dict:
    """
    Build a text-based dashboard summary for a given set of KPIs.

    Args:
        title: Dashboard title (e.g. 'Q1 2025 Performance').
        metrics_json: JSON object of metric_name → value pairs.

    Returns:
        dict with a formatted dashboard string.
    """
    try:
        metrics = json.loads(metrics_json)
        lines = [f"📊 DASHBOARD: {title}", "=" * 40]
        for k, v in metrics.items():
            lines.append(f"  • {k:<20} {v}")
        lines.append("=" * 40)
        return {"status": "success", "dashboard": "\n".join(lines)}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


# ── ADK Agent ─────────────────────────────────────────────────────────────────

analytics_agent = Agent(
    name="analytics_agent",
    model=oci_model,
    instruction="""You are a precise Analytics Expert assistant for:
- Growth analysis
- KPI metrics reports
- Trend detection

Rules:
- Always use tools for numeric analysis — never estimate.
- Present insights clearly: highlight the key takeaway first.
- Use emojis sparingly for trend direction (📈 📉 ➡️).
- If data is ambiguous, ask a clarifying question.
- For multi-metric requests, chain multiple tool calls.""",
    tools=[calculate_growth_rate, generate_metrics_report, analyze_trend, build_dashboard_summary],
)


def main() -> None:
    import uvicorn

    PORT = int(os.environ.get("ANALYTICS_AGENT_PORT", 8891))
    HOST = os.environ.get("AGENT_HOST", "0.0.0.0")

    a2a_app = to_a2a(analytics_agent, port=PORT)

    print(f"📊 Analytics Agent running at http://{HOST}:{PORT}")

    uvicorn.run(a2a_app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
