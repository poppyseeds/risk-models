from flask import Flask

from api.routes import create_api


class FakeInferenceService:
    def __init__(self):
        self.assets = {
            "ready": True,
            "errors": [],
            "network_model": object(),
            "network_scaler": object(),
            "process_model": object(),
            "process_scaler": object(),
            "hardware_model": object(),
            "hardware_scaler": object(),
        }

    def run(self, payload):
        return {
            "detected_at": "2026-04-15T00:00:00+00:00",
            "timestamp": payload["timestamp"],
            "site_id": payload["site_id"],
            "asset_id": payload["asset_id"],
            "system_status": "healthy",
            "fused_risk_score": 0.62,
            "network_score": 0.7,
            "process_score": 0.5,
            "score_breakdown": {},
            "network_normalized": 0.7,
            "process_normalized": 0.5,
            "severity": "medium",
            "anomaly_reason": "Correlated anomaly",
            "top_contributors": {"network": [], "process": []},
            "recommendation": "Inspect PLC configuration and verify recent command activity.",
            "model_status": {"ready": True, "errors": []},
        }


class FakeStore:
    def record_incident(self, result):
        return None

    def latest_incidents(self, limit=20):
        return []

    def clear_incidents(self):
        return 0


class FakeAIAnalystService:
    def status(self):
        return {
            "enabled": True,
            "ready": True,
            "provider": "ollama",
            "model": "llama3.1:8b",
            "detail": None,
            "available_models": ["llama3.1:8b"],
        }

    def analyze(self, payload, risk):
        return {
            "status": "available",
            "provider": "ollama",
            "model": "llama3.1:8b",
            "generated_at": "2026-04-15T00:00:00+00:00",
            "summary": "Likely coordinated activity affecting the PLC.",
            "what_happened": "An anomaly was detected in the control asset.",
            "where_it_happened": f"{payload['site_id']}/{payload['asset_id']}",
            "why_it_happened": "Network and process indicators were both elevated.",
            "evidence": ["network_score=0.7", "process_score=0.5"],
            "immediate_actions": ["Isolate the affected PLC."],
            "recovery_plan": ["Review configuration changes."],
            "confidence": "medium",
        }


def _payload():
    return {
        "site_id": "plant-a",
        "asset_id": "plc-01",
        "timestamp": "2026-04-15T00:00:00Z",
        "network": {
            "packet_rate": 100,
            "bytes_per_sec": 1000,
            "avg_packet_size": 100,
            "tcp_syn_rate": 1,
            "failed_login_rate": 0,
            "new_connection_rate": 2,
            "external_ip_ratio": 0.1,
            "dns_query_rate": 3,
        },
        "process_sequence": [
            {
                "reactor_temp_c": 72,
                "reactor_pressure_bar": 8.2,
                "valve_position_pct": 44,
                "motor_current_a": 12.2,
            }
        ],
    }


def test_detect_route_returns_single_risk_object():
    app = Flask(__name__)
    app.register_blueprint(create_api(FakeInferenceService(), FakeStore()), url_prefix="/api")
    client = app.test_client()

    response = client.post("/api/detect", json=_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert "risk" in body
    assert body["risk"]["severity"] == "medium"


def test_detect_route_rejects_bad_payload():
    app = Flask(__name__)
    app.register_blueprint(create_api(FakeInferenceService(), FakeStore()), url_prefix="/api")
    client = app.test_client()

    response = client.post("/api/detect", json={"foo": "bar"})
    assert response.status_code == 400


def test_history_limit_invalid_returns_400():
    app = Flask(__name__)
    app.register_blueprint(create_api(FakeInferenceService(), FakeStore()), url_prefix="/api")
    client = app.test_client()

    assert client.get("/api/history?limit=abc").status_code == 400
    assert client.get("/api/history?limit=0").status_code == 400
    assert client.get("/api/history?limit=1000").status_code == 400


def test_history_limit_valid():
    app = Flask(__name__)
    app.register_blueprint(create_api(FakeInferenceService(), FakeStore()), url_prefix="/api")
    client = app.test_client()

    r = client.get("/api/history?limit=50")
    assert r.status_code == 200
    assert r.get_json() == {"history": []}


def test_history_delete_clears_timeline():
    class StoreWithData(FakeStore):
        def clear_incidents(self):
            return 3

    app = Flask(__name__)
    app.register_blueprint(create_api(FakeInferenceService(), StoreWithData()), url_prefix="/api")
    client = app.test_client()

    r = client.delete("/api/history")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["cleared"] == 3


def test_detect_route_rejects_bad_numeric_payload():
    app = Flask(__name__)
    app.register_blueprint(create_api(FakeInferenceService(), FakeStore()), url_prefix="/api")
    client = app.test_client()

    payload = _payload()
    payload["network"]["packet_rate"] = "not-a-number"
    response = client.post("/api/detect", json=payload)

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "invalid_payload"


def test_status_includes_domain_readiness():
    app = Flask(__name__)
    service = FakeInferenceService()
    service.assets["hardware_model"] = None
    app.register_blueprint(create_api(service, FakeStore()), url_prefix="/api")
    client = app.test_client()

    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.get_json()
    assert body["domain_readiness"] == {
        "network": True,
        "process": True,
        "hardware": False,
    }


def test_detect_route_includes_ai_analysis_when_service_is_present():
    app = Flask(__name__)
    app.register_blueprint(
        create_api(FakeInferenceService(), FakeStore(), FakeAIAnalystService()),
        url_prefix="/api",
    )
    client = app.test_client()

    response = client.post("/api/detect", json=_payload())

    assert response.status_code == 200
    body = response.get_json()
    assert body["risk"]["ai_analysis"]["status"] == "available"
    assert body["risk"]["ai_analysis"]["where_it_happened"] == "plant-a/plc-01"


def test_status_includes_ai_analyst_status():
    app = Flask(__name__)
    app.register_blueprint(
        create_api(FakeInferenceService(), FakeStore(), FakeAIAnalystService()),
        url_prefix="/api",
    )
    client = app.test_client()

    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.get_json()
    assert body["ai_analyst"]["ready"] is True
    assert body["ai_analyst"]["model"] == "llama3.1:8b"
    assert body["ai_analyst"]["provider"] == "ollama"
