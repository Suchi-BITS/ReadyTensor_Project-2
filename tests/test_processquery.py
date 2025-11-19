# test_process_query.py
import time
import pytest
import sys
import os
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.evaluate import evaluate
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.supervisor import finops_team_supervisor_node
from schema.state import init_state
from templates.custom_template import AnswerTemplate
from utils.prompt_loader import get_active_prompt_set_and_log
import uuid

# Import the function from integrations/main.py
# from integrations.main import process_query, build_graph

def test_process_query__with_real_graph():
    # Step 1: Create real graph
    # graph = build_graph()

    # Step 2: Define input, expected values and retrieval context
    query = "what was the total production environment cost in april?"
    expected_output = "Based on the analysis, the total cost for your production environment in April was **$232,026.97**. This figure represents the aggregated `EFFECTIVECOST` for resources tagged as 'prod' or 'production' within that specific `CHARGEPERIODSTART`.\n\nUnderstanding your `EFFECTIVECOST` by `ENVIRONMENT` is a crucial aspect of the FinOps capability of `Cost Allocation`. By accurately allocating costs to specific environments like production, you gain visibility into where your cloud spend is occurring, which is foundational for effective `Cost Optimization` and `Forecasting`.\n\nDo you have any further questions about this cost, or would you like to explore the breakdown of this `EFFECTIVECOST` by `SERVICE` or `RESOURCE` within the production environment?"
    retrieval_context = [
  "The FOCUS v1.1 specification supports tagging dimensions such as `environment` using the Tags column, enabling cost allocation to logical environments like 'production', 'dev', or 'test'.",
  "To calculate environment-specific spend, filter records where the `Tags` column contains values like 'prod' or 'production', then group by `ChargePeriodStart` and sum the `EffectiveCost` metric.",
  "The `EffectiveCost` column represents the amortized true cost of usage, accounting for discounts and upfront payments, and is used for accurate spend tracking in FinOps reporting.",
  "Analyzing cloud spend by environment supports the FinOps capabilities of Cost Allocation, Forecasting, and Cost Optimization by giving teams visibility into how much each environment is consuming."]

    run_id = str(uuid.uuid4())
    state = init_state(query, run_id)
    # print(state)
     # Step 3: Run the query using actual pipeline
    #result = process_query(graph, query, run_id)
    result = finops_team_supervisor_node(state)
    actual_output = result.get("response", "")
    print(f'***********************\nActual Output:\n{actual_output}\n***********************')
    # Step 5: Define metrics
    correctness = GEval(
            name="correctness",
            evaluation_steps=[
                "Check if the data presented in 'actual output' logically answers the question in 'expected output'.",
                "Minor syntax issues are acceptable if the logic is correct."
                "Additional context or information in 'actual output' is acceptable if it does not contradict the 'expected output'."

            ],
            criteria="Evaluate the overall quality of the response.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.7
        )
    
    metrics = [
        correctness,
        AnswerRelevancyMetric(threshold=0.7,include_reason=True,evaluation_template=AnswerTemplate),
        FaithfulnessMetric(threshold=0.7, include_reason=True),

    ]
    active_set,prompt_versions = get_active_prompt_set_and_log()
     # Step 4: Prepare test case
    evaluate (
        identifier = "run-WITH-tags-prompt",
        test_cases = [LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
        tags=["test", "dataquery", "finops"],
        name="hyperparameter-test")],
        metrics=metrics,
        hyperparameters={"prompt_version_set" : prompt_versions}
    )
    
    # Step 6: Run assertions
    # assert_test(test_case=test_case, metrics=metrics)


if __name__ == "__main__":
    test_process_query__with_real_graph()


# deepeval test run test/test_single_turn.py
# deepeval recommend metrics