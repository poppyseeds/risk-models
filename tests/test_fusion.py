from riskscore import FusionEngine


def test_fusion_returns_risk_object():
    engine = FusionEngine()
    result = engine.fuse([0.1, 0.2, 0.3], 0.15)
    assert "risk_score" in result
    assert 0.0 <= result["risk_score"] <= 1.0
    assert "reason" in result
