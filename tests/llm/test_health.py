from pydantic_ai.models.test import TestModel as OfflineModel

from verbatim.llm.health import check_model


def test_check_model_returns_a_typed_result_without_calling_an_api():
    model = OfflineModel(custom_output_args={"sentiment": "mixed", "reason": "Kind staff, late."})

    check = check_model(model)

    assert check.sentiment == "mixed"
    assert check.model == "test"
    assert check.input_tokens > 0
    assert check.cost_usd is None  # an offline model has no price
