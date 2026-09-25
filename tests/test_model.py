import mlflow
import pytest
from mlflow.types.schema import Array

from tic_tac_toe_model import TicTacToeModel

EXAMPLE_STATE = (0, 0, 0, 0, 1, 0, 0, 0, 0)  # matches get_input_example()


@pytest.fixture
def model():
    return TicTacToeModel(
        q_values={EXAMPLE_STATE: {0: 0.1, 2: 0.9, 8: 0.3}},
        hyperparameters={"discount_factor": 0.9},
        training_config={"total_games": 100},
        meta_data={"model_name": "test_model", "model_version": "1.0"},
    )


def test_model_signature_builds(model):
    sig = model.get_model_signature()

    assert sig.inputs.input_names() == ["current_player", "game_state"]
    assert sig.outputs.input_names() == ["action", "q_value"]
    game_state = next(spec for spec in sig.inputs.inputs if spec.name == "game_state")
    assert isinstance(game_state.type, Array)


def test_log_load_predict_round_trip(model, tmp_path, monkeypatch):
    """Logging with get_model_signature()/get_input_example() must succeed, and the reloaded model must predict.

    Note: MLflow 3 currently prefers the signature inferred from predict()'s type hints over the explicit one,
    so this deliberately doesn't assert on the stored signature.
    """
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    model.save(tmp_path)

    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=model,
            artifacts=model.get_artifact_path(),
            signature=model.get_model_signature(),
            input_example=model.get_input_example(),
        )

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    assert loaded.predict(model.get_input_example()) == {"action": 2, "q_value": pytest.approx(0.9)}
