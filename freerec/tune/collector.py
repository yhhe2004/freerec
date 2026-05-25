import os
from typing import Any, Dict, Iterable, List, Optional

from freerec.utils import import_pickle


def metric_direction(metric: str):
    name = str(metric).upper()
    return min if "LOSS" in name or "ERROR" in name else max


def get_metric(metrics: Dict[str, Dict[str, Any]], metric: str) -> Optional[float]:
    for mode in ("valid", "best", "test", "train"):
        val = metrics.get(mode, {}).get(metric)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def sample_history(history: List[Any], points: int) -> List[Dict[str, Any]]:
    if not history:
        return []
    if points <= 0 or len(history) <= points:
        return [{"step": idx, "value": val} for idx, val in enumerate(history)]
    indices = sorted(set(round(i * (len(history) - 1) / (points - 1)) for i in range(points)))
    return [{"step": idx, "value": history[idx]} for idx in indices]


class TrialCollector:
    """Read existing run artifacts into tune-friendly records."""

    def __init__(
        self,
        data_dir: str,
        best_filename: str,
        monitor_filename: str,
        analyze_metric: Iterable[str],
        curve_sample_points: int = 10,
    ):
        self.data_dir = data_dir
        self.best_filename = best_filename
        self.monitor_filename = monitor_filename
        self.analyze_metric = list(analyze_metric or [])
        self.curve_sample_points = int(curve_sample_points or 10)

    def collect_trial(
        self, id_: str, log_path: str, params: Dict[str, Any], status: str
    ) -> Dict[str, Any]:
        data_path = os.path.join(log_path, self.data_dir)
        best_path = os.path.join(data_path, self.best_filename)
        monitor_path = os.path.join(data_path, self.monitor_filename)
        record = {
            "id": id_,
            "status": status,
            "params": dict(params),
            "log_path": log_path,
            "best_path": best_path,
            "monitor_path": monitor_path,
            "metrics": {},
            "curves": {},
        }
        if os.path.exists(best_path):
            record["metrics"] = dict(import_pickle(best_path))
        if os.path.exists(monitor_path):
            record["curves"] = self.collect_curves(monitor_path)
        return record

    def collect_curves(self, monitor_path: str) -> Dict[str, Any]:
        monitors = import_pickle(monitor_path)
        curves = {}
        valid = monitors.get("valid", {})
        for metric in self.analyze_metric:
            history = None
            for bucket in valid.values():
                if metric in bucket:
                    history = bucket[metric]
                    break
            if history is not None:
                curves[metric] = sample_history(history, self.curve_sample_points)
        return curves

    def select_best(self, trials: List[Dict[str, Any]], which4best: str) -> Dict[str, Any]:
        finished = [trial for trial in trials if trial.get("status") == "completed"]
        if not finished:
            return {}
        direction = metric_direction(which4best)
        scored = [
            (get_metric(trial.get("metrics", {}), which4best), trial)
            for trial in finished
        ]
        scored = [(score, trial) for score, trial in scored if score is not None]
        if not scored:
            return {}

        if direction is min:
            return min(scored, key=lambda item: item[0])[1]
        return max(scored, key=lambda item: item[0])[1]
