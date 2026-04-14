import numpy as np


class FusionEngine:
    """
    Expects network window values and process score already in [0, 1] (anomaly degree).
    Uses clipped values so one domain cannot dominate via unbounded normalization.
    """

    def fuse(self, net_scores_window, process_score):
        net = float(np.clip(np.mean(net_scores_window), 0.0, 1.0))
        proc = float(np.clip(float(process_score), 0.0, 1.0))

        # Order matters: coordinated only when both domains are clearly elevated (not one-sided).
        if net >= 0.70 and proc >= 0.70:
            risk = 0.95
            reason = "Coordinated attack (network + process)"

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
            risk = float(0.38 * net + 0.62 * proc)
            reason = "Low risk / normal variation"

        risk = float(np.clip(risk, 0.0, 1.0))

        return {
            "network_norm": net,
            "process_norm": proc,
            "risk_score": risk,
            "reason": reason,
        }