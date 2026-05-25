# Sequential Tune and Vistune

## Motivation

Manual hyperparameter tuning is repetitive: run one grid, inspect TensorBoard,
pick the best setting, then continue with the next parameter group. The goal is
to let `freerec tune` run this workflow sequentially, use the existing
`which4best` metric to choose the best trial, and optionally ask an LLM to
summarize whether the current search range is reliable or should be expanded.

The existing flat `params` format, log layout, TensorBoard output, and
`results.json` aggregation must continue to work.

## User Experience

### Tuning

Users can keep using the existing command:

```bash
freerec tune MF-BPR cfg.yaml
```

The config may now group `params` from `group1` to `group10`:

```yaml
command: python main.py

envs:
  root: ../../data
  dataset: Gowalla_550811_ROU
  device: '0,1,2,3'
  llm_analyzer: deepseek
  api_key: xxxxxxxx
  analyze_metric: [NDCG@10, NDCG@20]

params:
  group1:
    reg_weight: [0.1, 0.01, 0.001, 0.0001]
    lr: [1.e-2, 1.e-3, 1.e-4]
  group2:
    seed: [0, 1, 2, 3, 4]

defaults:
  config: configs/Gowalla_MF.yaml
```

`which4best` remains the single source of truth for selecting the best trial.
It can come from the default training config loaded by `main.py`, or be passed
through tune defaults:

```yaml
defaults:
  config: configs/Gowalla_MF.yaml
  which4best: NDCG@20
```

Execution flow:

1. Create one tune session under `logs/{description}/tune/{session_id}`.
2. Run `group1` grid search.
3. Collect each trial's existing `best.pkl` and `monitors.pkl`.
4. Select the best trial using `which4best`.
5. Sample valid curves for every `analyze_metric`.
6. Ask the configured LLM, currently only DeepSeek, whether the group should
   continue to the next group or expand the current search range.
7. If expansion is needed, append the suggested parameter values and rerun the
   current group.
8. Otherwise, freeze the group's best parameters and continue to the next group.
9. Write a final Markdown and JSON report.

Flat params remain valid and are treated as the original one-shot grid search
unless LLM-related envs are present. This protects the current workflow.

### Visualization

Users can open the latest session with:

```bash
freerec vistune MF-BPR
```

or a specific session:

```bash
freerec vistune MF-BPR --session 20260525-153012
```

The browser opens one long page. It is not a multi-page dashboard. The page
contains, top to bottom:

1. Session summary: status, dataset, current group, finished trials, best
   `which4best` value, and best parameters.
2. Group sections in order.
3. In each group:
   - a single best-metric chart containing all trials;
   - one compact table with params, best epoch if known, valid metrics, test
     metrics, log path, and status;
   - the LLM analysis;
   - an `Open TensorBoard` button for the corresponding log root.

## Log Layout

Existing paths stay intact:

```text
logs/{description}/{dataset}/{id}/
logs/{description}/core/
infos/{description}/core/
```

New path:

```text
logs/{description}/tune/{session_id}/
```

Session structure:

```text
logs/{description}/tune/20260525-153012/
├── manifest.json
├── state.json
├── groups/
│   ├── group1.json
│   └── group2.json
├── llm/
│   ├── group1.prompt.md
│   └── group1.response.md
└── reports/
    ├── report.md
    └── report.json
```

The tune session stores references to existing run artifacts instead of copying
them:

```text
logs/{description}/{dataset}/{id}/data/best.pkl
logs/{description}/{dataset}/{id}/data/monitors.pkl
logs/{description}/{dataset}/{id}/summary/SUMMARY.md
```

`api_key` is never persisted. If environment values are written to README or
JSON files, the key must be replaced by `***`.

## Implementation Architecture

Add a small package:

```text
freerec/tune/
├── __init__.py
├── session.py
├── planner.py
├── collector.py
├── analyzer.py
├── sequential.py
└── web.py
```

### `session.py`

Owns session directories and JSON state.

Responsibilities:

- create and load tune sessions;
- write `manifest.json`, `state.json`, and `groups/groupN.json`;
- write `reports/report.md` and `reports/report.json`;
- redact `api_key`.

### `planner.py`

Normalizes params and builds trial grids.

Responsibilities:

- detect grouped params;
- validate `group1` to `group10`;
- convert old flat params to an implicit single group when needed;
- combine `defaults`, frozen best params from previous groups, and current
  group params;
- apply numeric LLM expansion suggestions with a max-round guard.

### `collector.py`

Reads existing outputs.

Responsibilities:

- load `best.pkl`;
- load `monitors.pkl`;
- extract valid/test metrics;
- sample valid curves for `analyze_metric`;
- select best trial using `which4best`.

### `analyzer.py`

Runs optional LLM analysis.

Responsibilities:

- support `llm_analyzer: deepseek`;
- produce a strict JSON result;
- save prompt and response;
- fall back to deterministic local rules when LLM is disabled or fails.

Expected result:

```json
{
  "summary": "...",
  "decision": "continue",
  "suggested_params": {},
  "confidence": 0.8
}
```

`decision` is one of:

```text
continue
expand
stop
```

### `sequential.py`

Coordinates grouped tuning.

Responsibilities:

- reuse the existing `Adapter` subprocess execution and TensorBoard writing;
- run one group at a time;
- freeze best params before moving to the next group;
- persist progress after every trial and group;
- keep old `Adapter` behavior for non-grouped configs.

### `web.py`

Implements `freerec vistune`.

Responsibilities:

- find the latest or requested session;
- serve one static HTML page backed by session JSON;
- render all groups in a single scrollable page;
- provide an `Open TensorBoard` action.

## Compatibility Requirements

- Existing flat tune configs must keep working.
- Existing `logs/{description}/core/results.json` remains unchanged.
- Existing TensorBoard hparams output remains unchanged.
- Existing training scripts do not need code changes.
- `envs.llm_analyzer`, `envs.api_key`, and `envs.analyze_metric` are tune-only
  controls and must not be forwarded as CLI options to training subprocesses.
- `which4best` is reused; no new `best_metric` key is introduced.

## First Implementation Milestones

1. Add the spec and parser support for grouped params.
2. Add session, planner, collector, and deterministic analyzer foundations.
3. Add `SequentialTuner` and route grouped configs through it.
4. Add `freerec vistune` as a single-page local viewer.
5. Add tests for grouped parsing, grid planning, metric selection, and backward
   compatibility.
