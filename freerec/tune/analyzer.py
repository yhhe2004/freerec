import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

from .collector import get_metric


class TuneAnalyzer:
    """LLM-backed group analysis with a deterministic fallback."""

    def __init__(self, envs: Dict[str, Any]):
        self.provider = envs.get("llm_analyzer")
        self.api_key = envs.get("api_key")
        self.model = envs.get("llm_model", "deepseek-chat")

    def analyze(
        self,
        group_name: str,
        group_params: Dict[str, Any],
        trials: List[Dict[str, Any]],
        which4best: str,
        round_index: int,
        max_expand_rounds: int,
    ) -> Tuple[Dict[str, Any], str, str]:
        prompt = self.build_prompt(group_name, group_params, trials, which4best)
        if self.provider == "deepseek" and self.api_key:
            try:
                response = self.call_deepseek(prompt)
                return self.parse_response(response), prompt, response
            except Exception as exc:
                fallback = self.fallback(
                    group_params, trials, which4best, round_index, max_expand_rounds
                )
                fallback["summary"] = f"LLM analysis failed: {exc}. Used local rules."
                return fallback, prompt, json.dumps(fallback, indent=2)

        fallback = self.fallback(
            group_params, trials, which4best, round_index, max_expand_rounds
        )
        return fallback, prompt, json.dumps(fallback, indent=2)

    def build_prompt(
        self,
        group_name: str,
        group_params: Dict[str, Any],
        trials: List[Dict[str, Any]],
        which4best: str,
    ) -> str:
        payload = {
            "group": group_name,
            "which4best": which4best,
            "search_space": group_params,
            "trials": trials,
            "instructions": [
                "Judge whether valid metrics are stable and converged.",
                "Judge whether different hyperparameter values differ clearly.",
                "Judge whether numeric ranges show an increase-then-decrease pattern.",
                "If the best value is at a numeric boundary, suggest expanded values.",
                "Return strict JSON with summary, decision, suggested_params, confidence.",
            ],
        }
        return json.dumps(payload, indent=2, default=str)

    def call_deepseek(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You analyze hyperparameter tuning results and return strict JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps(body).encode("utf8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf8", errors="ignore")
            raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def parse_response(response: str) -> Dict[str, Any]:
        data = json.loads(response)
        decision = data.get("decision", "continue")
        if decision not in {"continue", "expand", "stop"}:
            data["decision"] = "continue"
        data.setdefault("summary", "")
        data.setdefault("suggested_params", {})
        data.setdefault("confidence", 0)
        return data

    def fallback(
        self,
        group_params: Dict[str, Any],
        trials: List[Dict[str, Any]],
        which4best: str,
        round_index: int,
        max_expand_rounds: int,
    ) -> Dict[str, Any]:
        if round_index >= max_expand_rounds:
            return {
                "summary": "Reached max expansion rounds; continue to the next group.",
                "decision": "continue",
                "suggested_params": {},
                "confidence": 0.5,
            }

        best = self.best_trial(trials, which4best)
        if not best:
            return {
                "summary": "No completed trial with the target metric was found.",
                "decision": "continue",
                "suggested_params": {},
                "confidence": 0.2,
            }

        suggestions = {}
        params = best.get("params", {})
        for name, values in group_params.items():
            vals = list(values if isinstance(values, (list, tuple)) else [values])
            if len(vals) < 2 or name not in params:
                continue
            try:
                numeric = [float(val) for val in vals]
                best_val = float(params[name])
            except (TypeError, ValueError):
                continue
            if best_val == numeric[0]:
                suggestions[name] = self.expand_left(numeric)
            elif best_val == numeric[-1]:
                suggestions[name] = self.expand_right(numeric)

        if suggestions:
            return {
                "summary": "Best value is on a numeric boundary; expand the search range.",
                "decision": "expand",
                "suggested_params": suggestions,
                "confidence": 0.6,
            }
        return {
            "summary": "Best value is not on a numeric boundary; continue to the next group.",
            "decision": "continue",
            "suggested_params": {},
            "confidence": 0.6,
        }

    @staticmethod
    def best_trial(trials: List[Dict[str, Any]], which4best: str) -> Dict[str, Any]:
        scored = [
            (get_metric(trial.get("metrics", {}), which4best), trial)
            for trial in trials
            if trial.get("status") == "completed"
        ]
        scored = [(score, trial) for score, trial in scored if score is not None]
        if not scored:
            return {}
        reverse = "LOSS" not in which4best.upper() and "ERROR" not in which4best.upper()
        return sorted(scored, key=lambda item: item[0], reverse=reverse)[0][1]

    @staticmethod
    def expand_left(values: List[float]) -> List[float]:
        first, second = values[0], values[1]
        if first == 0:
            step = second - first
            return [first - step * i for i in range(1, 4)]
        ratio = second / first if first else 10
        if ratio > 0 and ratio != 1:
            return [first / (ratio**i) for i in range(1, 4)]
        step = second - first
        return [first - step * i for i in range(1, 4)]

    @staticmethod
    def expand_right(values: List[float]) -> List[float]:
        prev, last = values[-2], values[-1]
        if prev != 0:
            ratio = last / prev
            if ratio > 0 and ratio != 1:
                return [last * (ratio**i) for i in range(1, 4)]
        step = last - prev
        return [last + step * i for i in range(1, 4)]
