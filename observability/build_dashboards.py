"""Generate the Grafana dashboards R1-R4 as JSON (dashboards as code, PLAN §6).

Run: python observability/build_dashboards.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "grafana" / "dashboards"
DATASOURCE = {"type": "prometheus", "uid": "prometheus"}


def panel(pid: int, title: str, expressions: list[tuple[str, str]], x: int, y: int, *,
          kind: str = "timeseries", unit: str = "short", w: int = 12, h: int = 8) -> dict:
    return {
        "id": pid, "type": kind, "title": title, "datasource": DATASOURCE,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "targets": [{"refId": chr(65 + i), "expr": expr, "legendFormat": legend, "datasource": DATASOURCE}
                    for i, (expr, legend) in enumerate(expressions)],
    }


def dashboard(uid: str, title: str, panels: list[dict]) -> dict:
    return {"uid": uid, "title": title, "schemaVersion": 39, "version": 1, "refresh": "5s",
            "time": {"from": "now-15m", "to": "now"}, "tags": ["smart-request-router"], "panels": panels}


S1_TOTAL = 'sum(rate(router_outcomes_total{topic=~"routed|escalations"}[1m]))'
DASHBOARDS = {
    "r1-overview": dashboard("r1-overview", "R1 · Overview", [
        panel(1, "Messages per second", [(S1_TOTAL, "msg/s")], 0, 0, unit="ops"),
        panel(2, "p95 processing latency (System 1)", [
            ("histogram_quantile(0.95, sum(rate(router_process_seconds_bucket[1m])) by (le))", "p95"),
            ("histogram_quantile(0.5, sum(rate(router_process_seconds_bucket[1m])) by (le))", "p50")],
            12, 0, unit="s"),
        panel(3, "System 1 coverage", [
            ('sum(rate(router_outcomes_total{topic="routed"}[1m])) / ' + S1_TOTAL, "coverage")],
            0, 8, kind="stat", unit="percentunit", w=6),
        panel(4, "Escalation rate by reason", [
            ("sum(rate(router_escalations_total[1m])) by (reason)", "{{reason}}")], 6, 8, w=18),
        panel(5, "DLQ messages", [('sum(increase(router_outcomes_total{topic="dlq"}[5m]))', "dlq")],
              0, 16, kind="stat", w=6),
    ]),
    "r2-routing-quality": dashboard("r2-routing-quality", "R2 · Routing Quality", [
        panel(1, "Route distribution", [
            ("sum(rate(router_route_messages_total[5m])) by (route)", "{{route}}")], 0, 0, w=24),
        panel(2, "Mean System 1 confidence", [
            ("rate(router_confidence_sum[5m]) / rate(router_confidence_count[5m])", "mean confidence")],
            0, 8, unit="percentunit"),
        panel(3, "Out-of-distribution escalations", [
            ('sum(rate(router_escalations_total{reason="ood"}[5m]))', "ood/s")], 12, 8, unit="ops"),
    ]),
    "r3-surges": dashboard("r3-surges", "R3 · Surges", [
        panel(1, "Volume per route (per minute)", [
            ("sum(increase(router_route_messages_total[1m])) by (route)", "{{route}}")], 0, 0, w=24, h=10),
        panel(2, "Surge alerts", [
            ("sum(increase(router_surge_alerts_total[1m])) by (route)", "{{route}}")], 0, 10, w=24),
    ]),
    "r4-system2": dashboard("r4-system2", "R4 · System 2", [
        panel(1, "Agent latency p95", [
            ("histogram_quantile(0.95, sum(rate(s2_latency_seconds_bucket[5m])) by (le))", "p95"),
            ("histogram_quantile(0.5, sum(rate(s2_latency_seconds_bucket[5m])) by (le))", "p50")],
            0, 0, unit="s"),
        panel(2, "Fallbacks to triage", [("sum(increase(s2_fallback_total[5m]))", "fallbacks")],
              12, 0, kind="stat", w=6),
        panel(3, "Tokens per minute", [("sum(rate(s2_tokens_total[1m])) * 60", "tokens/min")], 18, 0, w=6),
    ]),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in DASHBOARDS.items():
        (OUT / f"{name}.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(DASHBOARDS)} dashboards to {OUT}")


if __name__ == "__main__":
    main()
