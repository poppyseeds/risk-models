import numpy as np


class FusionEngine:
    """
    Expects network, process, and hardware scores already in [0, 1] (anomaly degree).
    Uses clipped values so one domain cannot dominate via unbounded normalization.
    """

    def fuse(self, net_scores_window, process_score, hardware_score=0.0, hardware_rule_hits=None):
        hardware_rule_hits = hardware_rule_hits or []
        net = float(np.clip(np.mean(net_scores_window), 0.0, 1.0))
        proc = float(np.clip(float(process_score), 0.0, 1.0))
        hw = float(np.clip(float(hardware_score), 0.0, 1.0))

        if hardware_rule_hits:
            summary = ", ".join(hardware_rule_hits[:2])
            if len(hardware_rule_hits) > 2:
                summary = f"{summary}, ..."
            risk = 1.0
            reason = f"CRITICAL: hardware tamper ({summary})"

        elif hw >= 0.82 and (net >= 0.48 or proc >= 0.48):
            risk = 0.99
            reason = "Coordinated attack with hardware compromise"

        elif hw >= 0.82:
            risk = 0.96
            reason = "Hardware anomaly (possible side-channel, glitch, or Trojan activity)"

        elif net >= 0.70 and proc >= 0.70:
            risk = 0.95
            reason = "Coordinated attack (network + process)"

        elif hw >= 0.60 and (net >= 0.48 or proc >= 0.48):
            risk = 0.88
            reason = "Multi-domain anomaly including hardware"

        elif hw >= 0.60:
            risk = 0.84 + 0.08 * hw
            reason = "Elevated hardware anomaly"

        elif proc >= 0.74 and net < 0.55:
            risk = 0.88
            reason = "Process anomaly (possible manipulation)"

        elif net >= 0.66 and proc < 0.55:
            risk = 0.78
            reason = "Network anomaly (possible intrusion)"

        elif net >= 0.48 and proc >= 0.48:
            risk = 0.62 + 0.08 * max(net, proc)
            reason = "Correlated anomaly (network and process both elevated)"

        else:
            risk = float(0.25 * net + 0.35 * proc + 0.40 * hw)
            reason = "Low risk / normal variation"

        risk = float(np.clip(risk, 0.0, 1.0))

        return {
            "network_norm": net,
            "process_norm": proc,
            "hardware_norm": hw,
            "risk_score": risk,
            "reason": reason,
        }
