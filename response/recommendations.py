def classify_severity(score: float) -> str:
    if score >= 0.9:
        return "critical"
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def recommendation_for_severity(severity: str) -> str:
    mapping = {
        "low": "Monitor traffic and process variables for trend change.",
        "medium": "Inspect PLC configuration and verify recent command activity.",
        "high": "Block suspicious IPs and segment the affected industrial device.",
        "critical": "Switch system to safe mode and isolate impacted control network immediately.",
    }
    return mapping.get(severity, mapping["low"])
