# CLAUDE.md — tic_tac_toe_model

This file is the authoritative guide for any LLM or automated tool working in this repository. Read it before editing any file.

---

## What this repo is

A minimal, standalone Python package. Its only export is `TicTacToeModel`, a class that:

1. Inherits `mlflow.pyfunc.PythonModel` — making it loggable to and loadable from MLflow.
2. Wraps a trained Tic Tac Toe Q-table and exposes a `predict` method for inference.
3. Handles two artifact formats: **safetensors** (this class's native format) and **pickle** (the format the training repo produces).

It has no training logic. It sits between the training repo and the serving repo as a shared contract.

---

## Repo layout

```
tic_tac_toe_model/          ← repo root
├── pyproject.toml          ← hatchling build; lists all runtime deps
├── README.md               ← human + LLM usage documentation
├── CLAUDE.md               ← this file
└── tic_tac_toe_model/      ← installable package
    ├── __init__.py         ← re-exports TicTacToeModel only
    └── model.py            ← TicTacToeModel (the entire package is this one class)
```

The package intentionally has exactly one class and one module. Do not add modules without a clear reason.

---

## The only class: TicTacToeModel

File: `tic_tac_toe_model/model.py`

### Constructor signature

```python
TicTacToeModel(
    q_values: Dict[Tuple[int, ...], Dict[int, float]],
    hyperparameters: Dict,
    training_config: Dict,
    meta_data: Dict,
)
```

All four arguments are **required**. Do not make them optional.

### Internal state

| Attribute | Type | Set by |
|-----------|------|--------|
| `self.q_values` | `Dict[Tuple, Dict[int, float]]` | constructor, `load()`, `load_context()` |
| `self.hyperparameters` | `Dict` | constructor |
| `self.training_config` | `Dict` | constructor |
| `self.meta_data` | `Dict` | constructor |
| `self.artifact_dir` | `Path \| None` | `save()`, `load()` |

`artifact_dir` starts as `None`. It is set when the model is saved or loaded from disk. `get_artifact_path()` raises `ValueError` if it is still `None`.

---

## Q-values contract

This is the central data structure. Every method that touches `self.q_values` must conform to it.

```
Type:  dict[tuple[int, ...], dict[int, float]]

Outer key:  9-tuple of ints, each 0/1/2
            index maps to board position (row-major):
              0 1 2
              3 4 5
              6 7 8
            value: 0=empty, 1=player1, 2=player2

Inner key:  int, board position 0–8 (the action)
Inner val:  float, Q-value for that action in that state
            higher = more desirable for the agent
            only visited actions are present (no sentinel for absent ones)
```

### Pickle format (from training repo)

`dict[tuple, dict[int, float]]` — exactly the in-memory format above. `pickle.load` gives you a usable dict with no conversion.

### Safetensors format (this class's native format)

Each state is one named tensor:
- **Key:** `str(state_tuple)` — the string representation of the tuple, e.g. `"(0, 1, 0, 2, 1, 0, 0, 0, 0)"`
- **Shape:** `(9,)` — fixed length, one slot per board position
- **dtype:** `float32`
- **Encoding:** `arr[action] = q_value`. Absent actions are `NaN`.

Loading converts back: `{i: float(v) for i, v in enumerate(arr) if not np.isnan(v)}`.

**Critical invariant:** The NaN encoding is the only way action indices survive the safetensors round-trip. Do not change the save/load logic without preserving this property.

---

## Method contracts

### `load_context(context)`

Called by MLflow on model load. Must not be called manually.

```
if "q_values_safetensors" in context.artifacts → load via safetensors → reconstruct dict[int, float] per state
elif "q_values_pickle" in context.artifacts   → load via pickle → already dict[int, float]
else → raise ValueError
```

Never add a silent fallback. The ValueError is intentional — a missing artifact key indicates a misconfigured MLflow run.

### `predict(context, model_input)`

```
model_input: List[Dict]  — always a list with exactly one element
  model_input[0]["game_state"]     : List[int], length 9, values 0/1/2
  model_input[0]["current_player"] : int, 1 or 2

returns: {"action": int, "q_value": float}
  both values are native Python types (not numpy scalars) — required for JSON serialisation
```

`context` is the MLflow context object. Pass `None` when calling directly (e.g. inside a FastAPI endpoint).

Raises `ValueError` on an unknown board state. There is no fallback to random moves in `predict` / `_get_action`. If you need a fallback, add it in the calling code.

### `_get_action(game_state: TicTacToe) -> Dict`

Private. Called only by `predict`. Returns `{"action": int, "q_value": float}`.

`int(best_action)` and `float(...)` casts are deliberate — they prevent numpy scalars from reaching the JSON serialiser.

### `_get_state(env: TicTacToe) -> tuple`

Private. Returns the 9-tuple board state from a `TicTacToe` object. Every int is explicitly cast to prevent numpy types leaking into dict keys.

### `save(save_folder_path: Path | None)`

Writes `Q_values.safetensors`. Passing `None` writes to the current working directory. Sets `self.artifact_dir`.

### `load(load_folder_path: Path | None)`

Reads `Q_values.safetensors`. Passing `None` reads from the current working directory. Sets `self.artifact_dir`.

### `get_artifact_path() -> Dict[str, str]`

Returns `{"q_values_safetensors": "<absolute_path>"}`. The key `"q_values_safetensors"` must match the key checked in `load_context`. Do not rename it.

### `get_model_signature() -> ModelSignature`

Returns the MLflow schema. Only change this if `predict`'s input/output schema changes.

### `get_model_uri() -> str`

Returns `"models:/<model_name>/<model_version>"` from `self.meta_data`. Pure string formatting — no MLflow calls.

---

## Key consistency rules

1. The artifact key `"q_values_safetensors"` appears in both `load_context` and `get_artifact_path`. If you rename one, rename both.
2. `save()` must always produce a (9,) float32 tensor with NaN for absent actions. `load()` and `load_context` must invert this exactly. The two are coupled.
3. `best_action` must be cast to `int()` before being returned from any method. Same for `q_value` → `float()`. Numpy scalars break FastAPI's JSON serialiser.
4. All four constructor parameters must remain required. The class is a contract; optional parameters introduce ambiguity about what a model instance contains.

---

## Dependencies

Declared in `pyproject.toml`. Do not add training-specific dependencies (e.g. `torch` for training loops, `matplotlib`, experiment tracking beyond what MLflow already provides).

| Package | Why it's here |
|---------|--------------|
| `mlflow` | `PythonModel` base class and `ModelSignature` |
| `numpy` | Board reshaping in `predict`, NaN arrays in `save` |
| `safetensors` | Artifact format |
| `tic_tac_toe_game` | `TicTacToe` used in `predict` and `_get_state` |

---

## Extension pattern

To support a new model type (DQN, policy gradient, etc.), subclass `TicTacToeModel`. Override only what changes. See `README.md` for a worked example.

Methods you will almost always override: `load_context`, `_get_action`, `save`, `load`.
Methods you will rarely need to override: `predict`, `_get_state`, `get_artifact_path`, `get_model_signature`, `get_model_uri`.

Do not modify `TicTacToeModel` itself to accommodate a new model type — subclass it instead.

---

## What not to do

- Do not add a `get_action(env: TicTacToe) -> int` convenience wrapper to this class. The serving repo adapts to this class's interface, not the other way around.
- Do not make constructor parameters optional. Callers must be explicit about what a model contains.
- Do not add a silent fallback (random move) inside `predict` or `_get_action`. Unknown states are an error condition — surface them.
- Do not store numpy types in the return value of `predict` or `_get_action`. Always cast.
- Do not import training-specific libraries at the module level.
