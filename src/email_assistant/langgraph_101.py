from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv("../.env", override=True)

llm = init_chat_model("openai:gpt-4.1", temperature=0)

@tool
def write_email(to:str, subject:str, content:str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

model_with_tools = llm.bind_tools([write_email], tool_choice="any", parallel_tool_calls=False)

from typing import Literal
from langgraph.graph import StateGraph, START, END, MessagesState

workflow = StateGraph(MessagesState)

def call_llm(state: MessagesState) -> MessagesState:
    output = model_with_tools.invoke(state["messages"])
    return {"messages": [output]}

def run_tool(state: MessagesState):
    result = []
    for tool_call in state["messages"][-1].tool_calls: # type: ignore
        observation = write_email.invoke(tool_call["args"])
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["run_tool", END]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls: # type: ignore
        return "run_tool"
    return END

workflow.add_node("call_llm", call_llm)
workflow.add_node("run_tool", run_tool)
workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges("call_llm", should_continue, {"run_tool": "run_tool", END: END})
workflow.add_edge("run_tool", END)

app = workflow.compile()

if __name__ == "__main__":
    result = llm.invoke("What is an agent?")
    print(f"`result` type: {type(result)}")

    print(f"`write_email` type: {type(write_email)}")
    print(f"`write_email` args:\n{write_email.args}")
    print(f"`write_email` description: {write_email.description}")

    output = model_with_tools.invoke("Draft a response to my boss (boss@company.ai) about tomorrow's meeting")
    print(f"`output` type: {type(output)}")
    print(output)

    args = output.tool_calls[0]['args']
    print(args)

    tool_result = write_email.invoke(args)
    print(tool_result)

    result = app.invoke({"messages": [{"role": "user", "content": "Draft a response to my boss (boss@company.ai) confirming that I want to attend Interrupt!"}]})
    
    for m in result["messages"]:
        m.pretty_print()
