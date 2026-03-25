import numpy as np

class FusionEngine:

    def __init__(self):
        # tune these later
        self.NET_MIN, self.NET_MAX = 0.0, 0.5
        self.PROC_MIN, self.PROC_MAX = 0.0, 0.3

    def normalize(self, score, min_val, max_val):
        return (score - min_val) / (max_val - min_val + 1e-8)

    def fuse(self, net_scores_window, process_score):

        # 1. Aggregate network scores
        net_score = np.mean(net_scores_window)

        # 2. Normalize
        net_norm = self.normalize(net_score, self.NET_MIN, self.NET_MAX)
        proc_norm = self.normalize(process_score, self.PROC_MIN, self.PROC_MAX)

        # 3. Context-aware logic
        if net_norm > 0.8 and proc_norm > 0.8:
            risk = 0.95
            reason = "Coordinated attack (network + process)"

        elif proc_norm > 0.85:
            risk = 0.9
            reason = "Process anomaly (possible manipulation)"

        elif net_norm > 0.85:
            risk = 0.75
            reason = "Network anomaly (possible intrusion)"

        elif net_norm > 0.5 and proc_norm > 0.5:
            risk = 0.7
            reason = "Correlated anomaly"

        else:
            # soft fusion
            risk = 0.4 * net_norm + 0.6 * proc_norm
            reason = "Low risk / normal variation"

        return {
            "network_norm": float(net_norm),
            "process_norm": float(proc_norm),
            "risk_score": float(risk),
            "reason": reason
        }
    


fusion = FusionEngine()

result = fusion.fuse(
    net_scores_window=[0.1, 0.2, 0.15, 0.4],  
    process_score=0.25
)

print(result)