import os
import signal
import time
from typing import Any, Dict, List

from freerec.launcher import Adapter
from freerec.utils import import_yaml, infoLogger, timemeter

from .analyzer import TuneAnalyzer
from .collector import TrialCollector, get_metric
from .planner import GroupPlanner
from .session import TuneSession


class SequentialTuner(Adapter):
    """Run grouped params in order while preserving Adapter trial execution."""

    def compile(self, cfg) -> None:
        self.cfg = cfg
        self.devices = tuple(str(cfg.ENVS.device).split(","))
        self.planner = GroupPlanner(cfg.PARAMS, cfg.DEFAULTS)
        self.which4best = self.resolve_which4best()
        self.cfg["which4best"] = self.which4best
        self.session = TuneSession.create(cfg.ENVS.description, cfg)
        analyze_metric = cfg.ENVS.get("analyze_metric", [])
        if isinstance(analyze_metric, str):
            analyze_metric = [analyze_metric]
        if not analyze_metric:
            analyze_metric = [self.which4best]
        curve_sample_points = int(cfg.ENVS.get("curve_sample_points", 10))
        self.max_expand_rounds = int(cfg.ENVS.get("max_expand_rounds", 3))
        self.collector = TrialCollector(
            cfg.DATA_DIR,
            cfg.MONITOR_BEST_FILENAME,
            cfg.MONITOR_FILENAME,
            analyze_metric,
            curve_sample_points,
        )
        self.analyzer = TuneAnalyzer(cfg.ENVS)
        self.cfg.COMMAND = self.build_command()
        self.group_states = []

    def resolve_which4best(self) -> str:
        if self.cfg.DEFAULTS.get("which4best") is not None:
            return self.cfg.DEFAULTS["which4best"]
        config = self.cfg.DEFAULTS.get("config")
        if config and os.path.exists(config):
            try:
                loaded = import_yaml(config)
                if loaded.get("which4best") is not None:
                    return loaded["which4best"]
            except Exception:
                pass
        return self.cfg.get("which4best", "LOSS")

    def build_command(self) -> str:
        command = self.cfg.COMMAND
        tune_only = set(getattr(self.cfg, "TUNE_ONLY_ENVS", ()))
        for key, val in self.cfg.ENVS.items():
            if key == "device" or key in tune_only:
                continue
            command += self.get_option(key, val)
        return command

    @timemeter
    def fit(self) -> None:
        fixed_params: Dict[str, Any] = {}
        status = "completed"

        def signal_handler(sig, frame):
            self.session.finish("interrupted")
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, signal_handler)

        try:
            for group_name, group_params in self.planner.groups.items():
                self.session.update_state(status="running", current_group=group_name)
                group_state = self.run_group(group_name, group_params, fixed_params)
                self.group_states.append(group_state)
                fixed_params.update(group_state.get("best_params", {}))
            self.write_report(fixed_params)
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except Exception:
            status = "failed"
            raise
        finally:
            self.session.finish(status)

    def run_group(
        self, group_name: str, group_params: Dict[str, Any], fixed_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        rounds = []
        all_trials = []
        current_params = group_params
        best_trial = {}
        analysis = {}

        for round_index in range(self.max_expand_rounds + 1):
            params_list = self.planner.product_grid(current_params, fixed_params)
            params_list = [
                params
                for params in params_list
                if params not in [trial.get("params") for trial in all_trials]
            ]
            trial_records = self.run_trials(params_list, group_name, all_trials, rounds)
            all_trials.extend(trial_records)
            best_trial = self.collector.select_best(all_trials, self.which4best)
            analysis, prompt, response = self.analyzer.analyze(
                group_name,
                current_params,
                all_trials,
                self.which4best,
                round_index,
                self.max_expand_rounds,
            )
            self.session.write_llm(group_name, prompt, response)

            round_state = {
                "round_id": round_index + 1,
                "params": current_params,
                "trials": trial_records,
                "analysis": analysis,
            }
            rounds.append(round_state)
            group_state = self.build_group_state(
                group_name, rounds, all_trials, best_trial, analysis
            )
            self.session.write_group(group_name, group_state)

            if analysis.get("decision") != "expand":
                return group_state
            current_params = self.planner.apply_suggestion(
                current_params, analysis.get("suggested_params", {})
            )

        return self.build_group_state(group_name, rounds, all_trials, best_trial, analysis)

    def run_trials(
        self,
        params_list: List[Dict[str, Any]],
        group_name: str,
        previous_trials: List[Dict[str, Any]],
        rounds: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        source = list(params_list)
        tasks = [None for _ in range(len(self.devices))]
        records: List[Dict[str, Any]] = []

        while source or any(task is not None for task in tasks):
            for index, task in enumerate(tasks):
                if task is None and source:
                    params = source.pop(0)
                    command, id_, log_path = self.register(self.devices[index])
                    process = self.run(command, params)
                    tasks[index] = (process, id_, log_path, params)
                    time.sleep(1)

            time.sleep(1)
            for index, task in enumerate(tasks):
                if task is None:
                    continue
                process, id_, log_path, params = task
                if process.poll() is None:
                    continue
                process.wait()
                self.write(id_, log_path, params)
                status = "completed" if process.returncode == 0 else "failed"
                record = self.collector.collect_trial(id_, log_path, params, status)
                records.append(record)
                group_state = self.build_group_state(
                    group_name,
                    rounds,
                    previous_trials + records,
                    self.collector.select_best(previous_trials + records, self.which4best),
                    {},
                )
                self.session.write_group(group_name, group_state)
                tasks[index] = None
        return records

    def build_group_state(
        self,
        group_name: str,
        rounds: List[Dict[str, Any]],
        trials: List[Dict[str, Any]],
        best_trial: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        best_params = {}
        best_value = None
        if best_trial:
            best_params = {
                key: val
                for key, val in best_trial.get("params", {}).items()
                if key not in self.planner.defaults
                or self.planner.defaults.get(key) != val
            }
            best_value = get_metric(best_trial.get("metrics", {}), self.which4best)
        return {
            "name": group_name,
            "status": "completed" if analysis else "running",
            "which4best": self.which4best,
            "rounds": rounds,
            "trials": trials,
            "best_trial_id": best_trial.get("id") if best_trial else None,
            "best_value": best_value,
            "best_params": best_params,
            "llm_analysis": analysis,
        }

    def write_report(self, final_params: Dict[str, Any]) -> None:
        report = {
            "final_params": final_params,
            "groups": self.group_states,
        }
        self.session.write_json(os.path.join("reports", "report.json"), report)
        lines = ["# Tune Report", "", f"- which4best: `{self.which4best}`", ""]
        lines.append("## Final Params")
        for key, val in final_params.items():
            lines.append(f"- `{key}`: `{val}`")
        for group in self.group_states:
            lines.extend(["", f"## {group['name']}"])
            lines.append(f"- best trial: `{group.get('best_trial_id')}`")
            lines.append(f"- best value: `{group.get('best_value')}`")
            lines.append("")
            lines.append(group.get("llm_analysis", {}).get("summary", ""))
        path = self.session.path("reports", "report.md")
        with open(path, "w", encoding="utf8") as fh:
            fh.write("\n".join(lines))
        infoLogger(f"[Tune] >>> Report saved to {path}")
