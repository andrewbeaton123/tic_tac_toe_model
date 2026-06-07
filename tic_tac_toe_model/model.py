import ast
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Union

import mlflow.pyfunc
import numpy as np
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from safetensors.numpy import load_file, save_file
from tic_tac_toe_game import TicTacToe


class TicTacToeModel(mlflow.pyfunc.PythonModel):

    def __init__(
        self,
        q_values: Dict,
        hyperparameters: Dict,
        training_config: Dict,
        meta_data: Dict,
    ):
        self.q_values = q_values
        self.hyperparameters = hyperparameters
        self.training_config = training_config
        self.meta_data = meta_data
        self.artifact_dir = None

    def load_context(self, context):
        if "q_values_safetensors" in context.artifacts:
            tensors = load_file(context.artifacts["q_values_safetensors"])
            self.q_values = {
                ast.literal_eval(k): {i: float(v) for i, v in enumerate(arr) if not np.isnan(v)}
                for k, arr in tensors.items()
            }
        elif "q_values_pickle" in context.artifacts:
            with open(context.artifacts["q_values_pickle"], "rb") as f:
                self.q_values = pickle.load(f)
        else:
            raise ValueError(
                "No recognised artifact key found in context.artifacts. "
                "Expected 'q_values_safetensors' or 'q_values_pickle'."
            )

    def predict(
        self,
        context,
        model_input: List[Dict[str, Union[int, List[int]]]],
    ) -> Dict:
        game_state = model_input[0].get("game_state", [0, 0, 0, 0, 0, 0, 0, 0, 0])
        current_player = model_input[0].get("current_player", 1)
        current_game = TicTacToe(current_player, np.reshape(game_state, (3, 3)))
        return self._get_action(current_game)

    def _get_state(self, env: TicTacToe) -> tuple:
        return tuple(int(x) for x in env.board.reshape(-1))

    def _get_action(self, game_state: TicTacToe) -> Dict:
        state_key = self._get_state(game_state)
        if state_key in self.q_values:
            actions_q_values = self.q_values[state_key]
            best_action = max(actions_q_values, key=actions_q_values.get)
            return {"action": int(best_action), "q_value": float(actions_q_values[best_action])}
        else:
            raise ValueError(f"Untrained game state encountered: {state_key}")

    def save(self, save_folder_path: Path | None):
        # Actions are board positions 0-8. Store as a fixed 9-element array so
        # action indices survive the round-trip (NaN = action not in Q-table).
        MAX_ACTIONS = 9
        tensors = {}
        for state, action_vals in self.q_values.items():
            arr = np.full(MAX_ACTIONS, np.nan, dtype=np.float32)
            for action, val in action_vals.items():
                arr[int(action)] = float(val)
            tensors[str(state)] = arr
        file_size = sum(t.nbytes for t in tensors.values())

        if save_folder_path:
            save_file(tensors, os.path.join(save_folder_path, "Q_values.safetensors"))
        else:
            save_file(tensors, "Q_values.safetensors")

        logging.info(
            "Saving Q-values model",
            extra={
                "filepath": save_folder_path,
                "num_states": len(tensors),
                "file_size_mb": file_size / (1024**2),
                "model_version": self.meta_data.get("model_version", "Model Version Not Specified"),
            },
        )
        self.artifact_dir = save_folder_path or Path(".")

    def load(self, load_folder_path: Path | None):
        filepath = os.path.join(load_folder_path or ".", "Q_values.safetensors")
        try:
            tensors = load_file(filepath)
            self.q_values = {
                ast.literal_eval(k): {i: float(v) for i, v in enumerate(arr) if not np.isnan(v)}
                for k, arr in tensors.items()
            }
            logging.info(
                "Loaded Q-values model",
                extra={
                    "filepath": filepath,
                    "num_states": len(self.q_values),
                    "model_version": self.meta_data.get("model_version", "Model Version Not Specified"),
                },
            )
            self.artifact_dir = load_folder_path or Path(".")
        except FileNotFoundError:
            logging.error(f"Q-values file not found at {filepath}")
            raise
        except Exception as e:
            logging.error(f"Error loading Q-values from {filepath}: {e}")
            raise

    def get_artifact_path(self) -> Dict[str, str]:
        if self.artifact_dir is None:
            raise ValueError("Model artifacts not available. Call save() or load() first.")
        return {"q_values_safetensors": os.path.join(str(self.artifact_dir), "Q_values.safetensors")}

    def get_model_signature(self) -> ModelSignature:
        input_schema = Schema([
            ColSpec("integer", "current_player"),
            ColSpec("integer", "game_state", shape=(9,)),
        ])
        output_schema = Schema([
            ColSpec("integer", "action"),
            ColSpec("float", "q_value"),
        ])
        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def get_input_example(self) -> List[Dict]:
        return [{"current_player": 1, "game_state": [0, 0, 0, 0, 1, 0, 0, 0, 0]}]

    def get_model_uri(self) -> str:
        model_name = self.meta_data.get("model_name", "tictactoe-agent")
        model_version = self.meta_data.get("model_version", "1.0")
        return f"models:/{model_name}/{model_version}"
