import pytest

from freerec.tune.collector import TrialCollector
from freerec.tune.planner import GroupPlanner, is_grouped_params


class TestGroupPlanner:
    def test_flat_params_become_group1(self):
        planner = GroupPlanner({"lr": [0.1, 0.01]}, {"config": "a.yaml"})
        assert list(planner.groups.keys()) == ["group1"]
        assert planner.groups["group1"] == {"lr": [0.1, 0.01]}

    def test_grouped_params_are_detected(self):
        assert is_grouped_params({"group1": {"lr": [1]}, "group2": {"seed": [0]}})

    def test_grouped_params_must_be_continuous(self):
        with pytest.raises(ValueError):
            GroupPlanner({"group1": {"lr": [1]}, "group3": {"seed": [0]}}, {})

    def test_product_grid_uses_defaults_and_fixed_params(self):
        planner = GroupPlanner({"group1": {"lr": [0.1, 0.01]}}, {"config": "a.yaml"})
        grid = planner.product_grid(planner.groups["group1"], {"reg": 1e-4})
        assert grid == [
            {"config": "a.yaml", "reg": 1e-4, "lr": 0.1},
            {"config": "a.yaml", "reg": 1e-4, "lr": 0.01},
        ]


class TestTrialCollector:
    def test_select_best_max_metric(self):
        collector = TrialCollector("data", "best.pkl", "monitors.pkl", [])
        trials = [
            {"status": "completed", "metrics": {"valid": {"NDCG@20": 0.1}}},
            {"status": "completed", "metrics": {"valid": {"NDCG@20": 0.2}}},
        ]
        assert collector.select_best(trials, "NDCG@20") is trials[1]

    def test_select_best_min_metric(self):
        collector = TrialCollector("data", "best.pkl", "monitors.pkl", [])
        trials = [
            {"status": "completed", "metrics": {"valid": {"LOSS": 0.2}}},
            {"status": "completed", "metrics": {"valid": {"LOSS": 0.1}}},
        ]
        assert collector.select_best(trials, "LOSS") is trials[1]
