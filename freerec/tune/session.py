import json
import os
import time
from typing import Any, Dict, Optional

from freerec.utils import mkdirs


def redact(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            key: "***" if str(key).lower() == "api_key" else redact(val)
            for key, val in data.items()
        }
    if isinstance(data, list):
        return [redact(val) for val in data]
    return data


def jsonable(data: Any) -> Any:
    if isinstance(data, dict):
        return {str(key): jsonable(val) for key, val in data.items()}
    if isinstance(data, (list, tuple)):
        return [jsonable(val) for val in data]
    if hasattr(data, "item"):
        try:
            return data.item()
        except Exception:
            return str(data)
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    return str(data)


class TuneSession:
    """File-backed state for a sequential tune session."""

    def __init__(self, root: str):
        self.root = root
        self.groups_dir = os.path.join(root, "groups")
        self.llm_dir = os.path.join(root, "llm")
        self.reports_dir = os.path.join(root, "reports")
        mkdirs(self.root, self.groups_dir, self.llm_dir, self.reports_dir)

    @classmethod
    def create(cls, description: str, cfg) -> "TuneSession":
        session_id = time.strftime("%Y%m%d-%H%M%S")
        root = cfg.TUNE_LOG_PATH.format(description=description, session_id=session_id)
        session = cls(root)
        manifest = {
            "session_id": session_id,
            "description": description,
            "dataset": cfg.ENVS.get("dataset", ""),
            "command": cfg.COMMAND,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "running",
            "which4best": cfg.DEFAULTS.get("which4best", cfg.get("which4best", "")),
            "analyze_metric": cfg.ENVS.get("analyze_metric", []),
            "llm_analyzer": cfg.ENVS.get("llm_analyzer", ""),
            "envs": redact(dict(cfg.ENVS)),
        }
        session.write_json("manifest.json", manifest)
        session.write_json("state.json", {"status": "running", "current_group": None})
        return session

    @classmethod
    def load(cls, description: str, session_id: Optional[str] = None) -> "TuneSession":
        root = os.path.join("logs", description, "tune")
        if session_id is None:
            sessions = [
                name
                for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))
            ]
            if not sessions:
                raise FileNotFoundError(f"No tune sessions found for {description}.")
            session_id = sorted(sessions)[-1]
        return cls(os.path.join(root, session_id))

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def write_json(self, relpath: str, data: Dict[str, Any]) -> None:
        path = self.path(relpath)
        mkdirs(os.path.dirname(path))
        with open(path, "w", encoding="utf8") as fh:
            json.dump(jsonable(redact(data)), fh, indent=2)

    def read_json(self, relpath: str) -> Dict[str, Any]:
        with open(self.path(relpath), "r", encoding="utf8") as fh:
            return json.load(fh)

    def update_state(self, **kwargs) -> None:
        state = {}
        path = self.path("state.json")
        if os.path.exists(path):
            state = self.read_json("state.json")
        state.update(kwargs)
        self.write_json("state.json", state)

    def write_group(self, name: str, data: Dict[str, Any]) -> None:
        self.write_json(os.path.join("groups", f"{name}.json"), data)

    def write_llm(self, name: str, prompt: str, response: str) -> None:
        with open(self.path("llm", f"{name}.prompt.md"), "w", encoding="utf8") as fh:
            fh.write(prompt)
        with open(self.path("llm", f"{name}.response.md"), "w", encoding="utf8") as fh:
            fh.write(response)

    def finish(self, status: str = "completed") -> None:
        manifest = self.read_json("manifest.json")
        manifest["status"] = status
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.write_json("manifest.json", manifest)
        self.update_state(status=status)
