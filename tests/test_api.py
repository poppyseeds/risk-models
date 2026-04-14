from flask import Flask

from api.routes import create_api


class FakeInferenceService:
    def __init__(self):
        self.assets = {"ready": True, "errors": []}

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
