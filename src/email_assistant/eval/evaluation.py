from pathlib import Path

import pytest
from langsmith import Client
from langsmith import testing as t
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from email_assistant.email_assistant import email_assistant
from email_assistant.eval.email_dataset import (email_inputs, email_names, examples_triage, expected_tool_calls, response_criteria_list, triage_outputs_list)
from email_assistant.eval.prompts import RESPONSE_CRITERIA_SYSTEM_PROMPT
from email_assistant.utils import extract_tool_calls, format_messages_string

load_dotenv()


@pytest.mark.langsmith
@pytest.mark.parametrize("email_input, expected_calls", [(email_inputs[0], expected_tool_calls[0]), (email_inputs[3], expected_tool_calls[3])])
def test_email_dataset_tool_calls(email_input, expected_calls):
    result = email_assistant.invoke({"email_input": email_input})
    extracted_tool_calls = extract_tool_calls(result["messages"])

    missing_calls = [call for call in expected_calls if call.lower() not in extracted_tool_calls]

    t.log_outputs(
        {
            "missing_calls": missing_calls,
            "extracted_tool_calls": extracted_tool_calls,
            "response": format_messages_string(result["messages"]),
        }
    )

    assert len(missing_calls) == 0


class CriteriaGrade(BaseModel):
    justification: str = Field(description="The justification for the grade and score, including specific examples from the response.")
    grade: bool = Field(description="Does the response meet the provided criteria?")

criteria_eval_llm = init_chat_model("openai:gpt-4o")
criteria_eval_structured_llm = criteria_eval_llm.with_structured_output(CriteriaGrade)

def create_response_test_cases():
    test_cases = []
    for email_input, email_name, criteria, triage_output, expected_calls in zip(email_inputs, email_names, response_criteria_list, triage_outputs_list, expected_tool_calls):
        if triage_output == "respond":
            test_cases.append((email_input, email_name, criteria, expected_calls))

    print(f"Created {len(test_cases)} test cases for emails requiring responses")
    return test_cases


@pytest.mark.langsmith(output_keys=["criteria"])
@pytest.mark.parametrize("email_input,email_name,criteria,expected_calls", create_response_test_cases())
def test_response_criteria_evaluation(email_input, email_name, criteria, expected_calls):
    t.log_inputs({"test": "test_response_criteria_evaluation", "email": email_name})

    result = email_assistant.invoke({"email_input": email_input})
    all_messages_str = format_messages_string(result["messages"])

    eval_result = criteria_eval_structured_llm.invoke(
        [
            {"role": "system", "content": RESPONSE_CRITERIA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""\n\n Response criteria: {criteria} \n\n Assistant's response: \n\n {all_messages_str} \n\n Evaluate whether the assistant's response meets the criteria and provide justification for your evaluation.""",
            },
        ]
    )

    t.log_outputs({"justification": eval_result.justification, "response": all_messages_str})

    assert eval_result.grade


def target_email_assistant(inputs: dict) -> dict:
    response = email_assistant.nodes["triage_router"].invoke({"email_input": inputs["email_input"]})
    return {"classification_decision": response.update["classification_decision"]}


def classification_evaluator(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["classification_decision"].lower() == reference_outputs["classification"].lower()


def run_triage_evaluation():
    client = Client()

    dataset_name = "E-mail Triage Evaluation"
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name, description="A dataset of e-mails and their triage decisions.")
        client.create_examples(dataset_id=dataset.id, examples=examples_triage)

    return client.evaluate(
        target_email_assistant,
        data=dataset_name,
        evaluators=[classification_evaluator],
        experiment_prefix="E-mail assistant workflow",
        max_concurrency=2,
    ) # type: ignore


def run_llm_judge_demo(index: int = 0):
    email_input = email_inputs[index]
    success_criteria = response_criteria_list[index]
    print("\nEmail Input:", email_input["subject"])
    print("Success Criteria:", success_criteria.strip())

    response = email_assistant.invoke({"email_input": email_input})
    all_messages_str = format_messages_string(response["messages"])

    eval_result = criteria_eval_structured_llm.invoke(
        [
            {"role": "system", "content": RESPONSE_CRITERIA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""\n\n Response criteria: {success_criteria} \n\n Assistant's response: \n\n {all_messages_str} \n\n Evaluate whether the assistant's response meets the criteria and provide justification for your evaluation.""",
            },
        ]
    )

    print(f"\nGrade: {eval_result.grade}")
    print(f"Justification: {eval_result.justification}")
    return eval_result


def save_graph_png(path: str | None = None) -> str:
    """Render the graph to a PNG via mermaid.ink; save the Mermaid source if that fails."""
    if path is None:
        path = str(Path(__file__).parent / "graph.png")
    graph = email_assistant.get_graph(xray=True)
    try:
        with open(path, "wb") as f:
            f.write(graph.draw_mermaid_png())
        return path
    except Exception as e:
        # mermaid.ink unreachable (offline): save the Mermaid source instead of failing.
        mmd_path = str(Path(path).with_suffix(".mmd"))
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(graph.draw_mermaid())
        print(f"PNG render failed ({e}); saved Mermaid source to {mmd_path}")
        return mmd_path


def main():
    print("=" * 72, "\nGRAPH ARCHITECTURE\n", "=" * 72, sep="")
    graph_path = save_graph_png()
    print(f"Graph architecture saved to: {graph_path}")

    print("\nExample test case:")
    print("  Email Input:", email_inputs[0]["subject"])
    print("  Expected Triage Output:", triage_outputs_list[0])
    print("  Expected Tool Calls:", expected_tool_calls[0])
    print("  Response Criteria:", response_criteria_list[0].strip())

    print("\n" + "=" * 72, "\nMETHOD 2 — TRIAGE evaluate()\n", "=" * 72, sep="")
    run_triage_evaluation()

    print("\n" + "=" * 72, "\nMETHOD 3 — LLM-AS-JUDGE (demo)\n", "=" * 72, sep="")
    run_llm_judge_demo(index=0)

    print(
        "\nRun the pytest-based methods (tool calls + LLM-judge over the suite) with:\n"
        "    pytest src/email_assistant/eval/evaluation.py"
    )


if __name__ == "__main__":
    main()
