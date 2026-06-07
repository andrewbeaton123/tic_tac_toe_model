# tic_tac_toe_model

A standalone Python package that wraps a trained Tic Tac Toe Q-table as an [MLflow PythonModel](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html). It is the shared model contract between the training repo ([tic_tac_learn](https://github.com/andrewbeaton123/tic_tac_learn)) and the serving repo ([tic_tac_toe_model_serve](https://github.com/andrewbeaton123/tic_tac_toe_model_serve)).

---

## Table of contents

- [System overview](#system-overview)
- [Installation](#installation)
- [Q-values data structure](#q-values-data-structure)
- [Usage: start to end](#usage-start-to-end)
  - [1. After training — wrap the model](#1-after-training--wrap-the-model)
  - [2. Save to disk (safetensors)](#2-save-to-disk-safetensors)
  - [3. Load from disk (safetensors)](#3-load-from-disk-safetensors)
  - [4. Load from a training-repo pickle](#4-load-from-a-training-repo-pickle)
  - [5. Log to MLflow](#5-log-to-mlflow)
  - [6. Load from the MLflow registry](#6-load-from-the-mlflow-registry)
  - [7. Run inference](#7-run-inference)
- [API reference](#api-reference)
- [Artifact file formats](#artifact-file-formats)
- [Extending to new model types](#extending-to-new-model-types)

---

## System overview

```
┌─────────────────────────────┐
│   tic_tac_learn (training)  │
│                             │
│  trains Q-table             │
│  saves → saved_q_values.pkl │
└────────────┬────────────────┘
             │  pickle file
             ▼
┌─────────────────────────────┐
│   tic_tac_toe_model  ◄───── this repo
│                             │
│  TicTacToeModel             │
│   · wraps q_values          │
│   · save()  → safetensors   │
│   · load()  ← safetensors   │
│   · load_context() ← either │
│   · predict() → action+q    │
│   · MLflow logging helpers  │
└────────────┬────────────────┘
             │  MLflow registry or
             │  safetensors / pickle file
             ▼
┌─────────────────────────────┐
│  tic_tac_toe_model_serve    │
│  (FastAPI serving)          │
│                             │
│  loads model                │
│  POST /next_move → action   │
└─────────────────────────────┘
```

This package has no training logic. Its only job is to be the stable interface that both sides of the system import.

---

## Installation

**From this repository (recommended during development):**

```bash
pip install -e "git+https://github.com/andrewbeaton123/tic_tac_toe_model.git#egg=tic_tac_toe_model"
```

**Locally (editable install):**

```bash
git clone https://github.com/andrewbeaton123/tic_tac_toe_model.git
cd tic_tac_toe_model
pip install -e .
```

**Requirements:** Python ≥ 3.10. The following are installed automatically:

| Dependency | Purpose |
|------------|---------|
| `mlflow` | `PythonModel` base class and logging helpers |
| `numpy` | Board array manipulation |
| `safetensors` | Primary on-disk artifact format |
| `tic_tac_toe_game` | `TicTacToe` game environment used in inference |

---

## Q-values data structure

Everything in this package depends on understanding one data structure.

### Type

```python
Dict[
    Tuple[int, ...],   # board state — 9-tuple of cell values
    Dict[int, float]   # action → Q-value mapping
]
```

### Board state key

A 9-element tuple representing the board read left-to-right, top-to-bottom:

```
Board positions:
 0 | 1 | 2
-----------
 3 | 4 | 5
-----------
 6 | 7 | 8

Cell values:
  0 = empty
  1 = player 1
  2 = player 2
```

Example: the board below is encoded as `(0, 1, 0, 2, 1, 0, 0, 0, 0)`:

```
 _ | X | _
-----------
 O | X | _
-----------
 _ | _ | _
```

### Action mapping

The inner dict maps a board position index (0–8) to its Q-value (higher = more desirable for the agent):

```python
{
    2: 0.91,   # placing at position 2 is the best move
    5: 0.43,
    6: 0.21,
    7: 0.08,
    8: -0.12,
}
```

Not every action appears in the inner dict — only positions that were visited during training.

### Full example

```python
q_values = {
    (0, 0, 0, 0, 0, 0, 0, 0, 0): {0: 0.1, 1: 0.3, 4: 0.9, 8: 0.2},
    (1, 0, 0, 0, 0, 0, 0, 0, 0): {1: 0.4, 4: 0.7, 8: 0.1},
    # ... thousands more states
}
```

---

## Usage: start to end

### 1. After training — wrap the model

Once training is complete you have a `q_values` dict in memory. Wrap it:

```python
from tic_tac_toe_model import TicTacToeModel

model = TicTacToeModel(
    q_values=q_values,          # dict[tuple, dict[int, float]] from training
    hyperparameters={
        "learning_rate": 0.1,
        "discount_factor": 0.99,
        "epsilon": 0.1,
    },
    training_config={
        "episodes": 100_000,
        "opponent_type": "random",
    },
    meta_data={
        "model_name": "tictactoe-agent",   # used by get_model_uri()
        "model_version": "1.0",            # used by get_model_uri()
        "author": "andrew",
        "training_date": "2025-04-11",
        "win_rate_vs_random": 0.98,
    },
)
```

All four constructor arguments are required. They travel with the instance and are accessible as `model.hyperparameters`, `model.training_config`, and `model.meta_data`.

---

### 2. Save to disk (safetensors)

```python
from pathlib import Path

model.save(save_folder_path=Path("./artifacts"))
# writes: ./artifacts/Q_values.safetensors
```

Pass `None` to write to the current directory:

```python
model.save(save_folder_path=None)
# writes: ./Q_values.safetensors
```

After saving, `model.get_artifact_path()` is available:

```python
model.get_artifact_path()
# {"q_values_safetensors": "/abs/path/to/artifacts/Q_values.safetensors"}
```

---

### 3. Load from disk (safetensors)

```python
model = TicTacToeModel(
    q_values={},           # placeholder — overwritten by load()
    hyperparameters={},
    training_config={},
    meta_data={"model_version": "1.0"},
)
model.load(load_folder_path=Path("./artifacts"))
# reads: ./artifacts/Q_values.safetensors
# model.q_values is now populated
```

---

### 4. Load from a training-repo pickle

The [tic_tac_learn](https://github.com/andrewbeaton123/tic_tac_learn) training repo saves `dict(master_q_table)` as a pickle. Load it directly and pass it to the constructor:

```python
import pickle

with open("saved_q_values.pkl", "rb") as f:
    q_values = pickle.load(f)

model = TicTacToeModel(
    q_values=q_values,
    hyperparameters={},
    training_config={},
    meta_data={"model_name": "tictactoe-agent", "model_version": "1.0"},
)
```

The pickle file contains `dict[tuple[int,...], dict[int, float]]` exactly — no conversion needed.

---

### 5. Log to MLflow

```python
import mlflow

mlflow.set_experiment("tictactoe-training")

with mlflow.start_run():
    # Log hyperparameters and training config
    mlflow.log_params(model.hyperparameters)
    mlflow.log_params(model.training_config)

    # Save artifact to disk first, then log to MLflow
    artifact_dir = Path("./mlflow_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    model.save(artifact_dir)

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model,
        artifacts=model.get_artifact_path(),
        signature=model.get_model_signature(),
        input_example=model.get_input_example(),
    )
```

`get_artifact_path()` returns `{"q_values_safetensors": "<absolute_path>"}`. MLflow copies the file into the run's artifact store and passes the key back through `load_context` when the model is reloaded.

---

### 6. Load from the MLflow registry

```python
import mlflow.pyfunc

# By run ID
model_uri = "runs:/<run_id>/model"

# Or from the registry using the model's own URI helper
# model_uri = model.get_model_uri()   # "models:/tictactoe-agent/1.0"

loaded = mlflow.pyfunc.load_model(model_uri)
```

When `load_model` is called, MLflow reconstructs the object and calls `load_context`. That method checks which artifact key is present (`"q_values_safetensors"` or `"q_values_pickle"`) and populates `self.q_values` accordingly.

---

### 7. Run inference

**Via the raw class (e.g. inside a FastAPI endpoint):**

```python
model_input = [
    {
        "current_player": 1,
        "game_state": [0, 0, 0, 0, 1, 0, 0, 0, 0],
        #               pos 0           pos 4 (centre, player 1)
    }
]

result = model.predict(context=None, model_input=model_input)
# {"action": 2, "q_value": 0.914}
```

**Via MLflow's loaded wrapper:**

```python
result = loaded.predict(model_input)
# same output
```

`model_input` is always a list containing exactly one dict. The dict has two keys:

| Key | Type | Description |
|-----|------|-------------|
| `current_player` | `int` | `1` or `2`. Defaults to `1` if omitted. |
| `game_state` | `List[int]` (length 9) | Flat board, values 0/1/2. Defaults to all-zero if omitted. |

The response is always:

| Key | Type | Description |
|-----|------|-------------|
| `action` | `int` | Board position (0–8) of the recommended move |
| `q_value` | `float` | Q-value of that action for the given state |

---

## API reference

### `TicTacToeModel(q_values, hyperparameters, training_config, meta_data)`

Inherits from `mlflow.pyfunc.PythonModel`. All four arguments are required.

| Argument | Type | Description |
|----------|------|-------------|
| `q_values` | `Dict[Tuple, Dict[int, float]]` | The trained Q-table |
| `hyperparameters` | `Dict` | Training hyperparameters (logged to MLflow) |
| `training_config` | `Dict` | Run configuration (episodes, opponent type, etc.) |
| `meta_data` | `Dict` | Model metadata. Must contain `"model_name"` and `"model_version"` for registry methods to work correctly |

---

### `load_context(context)`

Called automatically by MLflow when loading a saved model. Do not call directly.

Checks `context.artifacts` for one of two keys:

| Key | Loader | Source |
|-----|--------|--------|
| `"q_values_safetensors"` | `safetensors.numpy.load_file` | Saved by `save()` from this class |
| `"q_values_pickle"` | `pickle.load` | Saved by the tic_tac_learn training repo |

Raises `ValueError` if neither key is present.

---

### `predict(context, model_input) -> Dict`

The MLflow inference entrypoint.

- **`context`** — MLflow context object. Pass `None` when calling directly.
- **`model_input`** — `List[Dict]` with exactly one element. See [inference section](#7-run-inference) for the schema.
- **Returns** `{"action": int, "q_value": float}`
- **Raises** `ValueError` if the board state has no entry in the Q-table.

---

### `save(save_folder_path: Path | None)`

Saves `self.q_values` to `Q_values.safetensors` in the given folder (or the current directory if `None`).

Sets `self.artifact_dir`, which enables `get_artifact_path()`.

---

### `load(load_folder_path: Path | None)`

Loads `Q_values.safetensors` from the given folder (or current directory) and overwrites `self.q_values`.

Sets `self.artifact_dir`.

---

### `get_artifact_path() -> Dict[str, str]`

Returns `{"q_values_safetensors": "<absolute_path_to_Q_values.safetensors>"}`.

Must be called after `save()` or `load()`. Raises `ValueError` otherwise.

Used as the `artifacts` argument to `mlflow.pyfunc.log_model`.

---

### `get_model_signature() -> ModelSignature`

Returns the MLflow `ModelSignature` describing the predict interface:

```
Inputs:
  current_player : integer
  game_state     : integer[9]

Outputs:
  action  : integer
  q_value : float
```

---

### `get_input_example() -> List[Dict]`

Returns a minimal valid input for `predict`, used by MLflow for documentation:

```python
[{"current_player": 1, "game_state": [0, 0, 0, 0, 1, 0, 0, 0, 0]}]
```

---

### `get_model_uri() -> str`

Returns the MLflow Model Registry URI for this model:

```
models:/<model_name>/<model_version>
```

Both values come from `self.meta_data`. Defaults to `models:/tictactoe-agent/1.0` if the keys are absent.

---

## Artifact file formats

### Safetensors (`Q_values.safetensors`)

The native format for this class. Produced and consumed by `save()` / `load()` / `load_context`.

Each board state is stored as one named tensor:

| Field | Value |
|-------|-------|
| Key | `str(state_tuple)` e.g. `"(0, 1, 0, 2, 1, 0, 0, 0, 0)"` |
| Shape | `(9,)` — one element per board position |
| dtype | `float32` |
| Encoding | `arr[action] = q_value`. Positions not in the Q-table for that state are `NaN`. |

Example tensor for state `(0, 1, 0, 2, 1, 0, 0, 0, 0)` with actions `{2: 0.91, 5: 0.43}`:

```
index: 0     1     2     3     4     5     6     7     8
value: NaN   NaN   0.91  NaN   NaN   0.43  NaN   NaN   NaN
```

The NaN encoding is what lets action indices survive the round-trip correctly.

### Pickle (`.pkl`)

Produced by the [tic_tac_learn](https://github.com/andrewbeaton123/tic_tac_learn) training repo. Contains a plain `dict` (converted from `defaultdict`) with the structure:

```python
dict[tuple[int, ...], dict[int, float]]
```

This package reads it directly via `pickle.load` in `load_context` — no conversion is needed because the in-memory structure is identical to what `save()` / `load()` reconstruct from safetensors.

---

## Extending to new model types

This class is designed as a base for other reinforcement learning methods (DQN, policy gradient, MCTS, etc.). The split between the **inference interface** and the **MLflow lifecycle interface** is intentional.

### What to override

| Method | When to override |
|--------|-----------------|
| `load_context` | When your artifact is not a Q-table (e.g. neural network weights) |
| `predict` | When your input/output schema changes |
| `_get_state` | When state representation changes |
| `_get_action` | When the decision logic changes |
| `save` / `load` | When your artifact format changes |
| `get_model_signature` | When the MLflow schema changes |

### What to keep

`get_artifact_path`, `get_input_example`, `get_model_uri` are mechanical MLflow helpers that only depend on `self.artifact_dir` and `self.meta_data`. They rarely need to change.

### Example subclass (DQN)

```python
import torch
from tic_tac_toe_model import TicTacToeModel

class TicTacToeDQN(TicTacToeModel):

    def __init__(self, network, hyperparameters, training_config, meta_data):
        # Pass an empty dict for q_values — not used by this subclass
        super().__init__({}, hyperparameters, training_config, meta_data)
        self.network = network

    def load_context(self, context):
        weights_path = context.artifacts["dqn_weights"]
        self.network.load_state_dict(torch.load(weights_path))
        self.network.eval()

    def _get_action(self, game_state) -> Dict:
        state_tensor = torch.tensor(self._get_state(game_state), dtype=torch.float32)
        with torch.no_grad():
            q_values = self.network(state_tensor)
        best_action = int(q_values.argmax())
        return {"action": best_action, "q_value": float(q_values[best_action])}
```

The constructor signature, `predict`, `get_model_signature`, and all registry helpers are inherited unchanged.
