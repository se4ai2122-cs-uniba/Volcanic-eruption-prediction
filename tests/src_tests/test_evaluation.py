import json
import pytest, sys
from pathlib import Path
sys.path.insert(1, str((Path(__file__).parent.parent.parent).resolve()))     #path of the project working directory relative to this file
from src.evaluate import evaluate_model

#check if the evaluation script correctly create the metrics file and if the metric decrease
@pytest.mark.evaluation
def test_model_evaluation():
    scores_file_path = Path("metrics") / "scores.json"
    if scores_file_path.exists():
        with open(scores_file_path, "r") as f:
            previous_rmse = json.load(f)['RMSE']
        assert type(previous_rmse) == float
        evaluate_model()
        with open(scores_file_path, "r") as f:
            new_rmse = json.load(f)['RMSE']
        assert type(new_rmse) == float
        assert new_rmse <= previous_rmse
    else:  # the metric file doesn't already exist
        evaluate_model()
        assert scores_file_path.exists()       #check if evaluate_model create the metric file
        with open(scores_file_path, "r") as f:
             assert type(json.load(f)['RMSE']) == float

