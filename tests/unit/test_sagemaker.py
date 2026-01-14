import json

import pytest

from preference_dynamics.sagemaker import inference_cnn1d_residual as inference


class TestSageMakerInference:
    def test_input_fn_single_dict(self) -> None:
        single_dict = {"time_series": [[1.0, 2.0], [3.0, 4.0]], "features": [True, 1.5, 2.5]}
        single_json = json.dumps(single_dict)
        result = inference.input_fn(single_json, "application/json")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == single_dict

    def test_input_fn_list_dicts(self) -> None:
        list_dicts = [
            {"time_series": [[1.0, 2.0], [3.0, 4.0]], "features": [True, 1.5, 2.5]},
            {"time_series": [[5.0, 6.0], [7.0, 8.0]], "features": [False, 0.0, 0.0]},
        ]
        list_json = json.dumps(list_dicts)
        result = inference.input_fn(list_json, "application/json")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_input_fn_invalid_content_type(self) -> None:
        with pytest.raises(ValueError, match="application/json"):
            inference.input_fn("{}", "text/plain")

    def test_output_fn_valid(self) -> None:
        predictions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        response, content_type = inference.output_fn(predictions, "application/json")

        assert isinstance(response, str)
        assert content_type == "application/json"
        parsed = json.loads(response)
        assert parsed == predictions

    def test_output_fn_invalid_accept(self) -> None:
        predictions = [[1.0, 2.0, 3.0]]
        with pytest.raises(ValueError, match="application/json"):
            inference.output_fn(predictions, "text/plain")
