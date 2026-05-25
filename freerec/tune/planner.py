from itertools import product
from typing import Any, Dict, Iterable, List, OrderedDict as OrderedDictType
from collections import OrderedDict


def is_group_name(name: str) -> bool:
    return name.startswith("group") and name[5:].isdigit()


def is_grouped_params(params: Dict[str, Any]) -> bool:
    return bool(params) and all(is_group_name(str(key)) for key in params.keys())


class GroupPlanner:
    """Normalize grouped tune params and build per-group trial grids."""

    MAX_GROUPS = 10

    def __init__(self, params: Dict[str, Any], defaults: Dict[str, Any]):
        self.groups = self.normalize_params(params)
        self.defaults = dict(defaults)

    @classmethod
    def normalize_params(cls, params: Dict[str, Any]) -> OrderedDictType[str, Dict]:
        params = dict(params or {})
        if not params:
            return OrderedDict()

        if not is_grouped_params(params):
            return OrderedDict([("group1", params)])

        groups = OrderedDict()
        indices = sorted(int(str(name)[5:]) for name in params.keys())
        expected = list(range(1, len(indices) + 1))
        if indices != expected:
            raise ValueError("Grouped params must be continuous from group1.")
        if len(indices) > cls.MAX_GROUPS:
            raise ValueError("Grouped params supports at most group10.")

        for idx in indices:
            name = f"group{idx}"
            group = dict(params[name] or {})
            if not group:
                raise ValueError(f"{name} must contain at least one parameter.")
            groups[name] = group
        return groups

    @staticmethod
    def ensure_list(vals: Any) -> List[Any]:
        if isinstance(vals, (str, int, float, bool)) or vals is None:
            return [vals]
        return list(vals)

    def product_grid(
        self, group_params: Dict[str, Iterable], fixed_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        names = list(group_params.keys())
        values = [self.ensure_list(group_params[name]) for name in names]
        trials = []
        for vals in product(*values):
            trial = dict(self.defaults)
            trial.update(fixed_params)
            trial.update({name: val for name, val in zip(names, vals)})
            trials.append(trial)
        return trials

    def apply_suggestion(
        self, group_params: Dict[str, Any], suggested_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        updated = {key: self.ensure_list(vals) for key, vals in group_params.items()}
        for key, vals in (suggested_params or {}).items():
            if key not in updated:
                continue
            for val in self.ensure_list(vals):
                if val not in updated[key]:
                    updated[key].append(val)
        return updated
