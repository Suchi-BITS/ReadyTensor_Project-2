import os
import sys
from langgraph.types import Send
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing_extensions import TypedDict,Annotated
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import InjectedToolCallId
from utils.prompt_loader import load_prompt_from_hub

datafetcher_task_desc = load_prompt_from_hub("transfer_to_datafetcher")
insightsagent_task_desc = load_prompt_from_hub("transfer_to_insightsagent")

@tool("transfer_to_data",description = "Assign a task to DataFetcher")
def transfer_to_data(
    task_description: Annotated[
        str,
        datafetcher_task_desc
    ],
    state: Annotated[MessagesState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    # print(task_description)
    tool_message = ToolMessage(
        content="Task successfully delegated to DataFetcherAgent.",
        name="transfer_to_insightsagent",
        tool_call_id=tool_call_id,
    )
    print("\n",tool_message.content,"\n")
    print("\n",task_description,"\n")
    # The new message for the DataFetcher agent contains only the structured task.
    task_description_message = {"role": "user", "content": task_description}
    # agent_input = {**state, "messages": [task_description_message]}
    print("DELEGATION FORMAT\n")
    print(task_description_message)
    print("DELEGATION FORMAT\n")
    return Command(
        goto="DataFetcher",
        graph = Command.PARENT,
        update={
            # Append the tool message to the supervisor's history
            "messages": state["messages"] + [tool_message,task_description_message]
            # Overwrite the 'messages' for the next agent with the specific task.
            # This is a key concept: the sub-agent receives a clean, direct instruction.
        },

    )

@tool(
    "transfer_to_insights",
    description="Assign a task to InsightsAgent"
)
def transfer_to_insights(
    task_description: Annotated[
        str,
        insightsagent_task_desc
    ],
    state: Annotated[MessagesState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
        
    # print("\n",task_description,"\n")
    tool_message = ToolMessage(
        content="Task successfully delegated to InsightAgent for Analysis",
        name = "transfer_to_insights",
        tool_call_id = tool_call_id
    )
    print(tool_message.content)
    print("\n",task_description,"\n")
    insights_description_message = {"role": "user", "content": task_description}
    # agent_input = {**state, "messages": [insights_description_message]}
    return Command(
        goto="InsightsAgent",
        graph=Command.PARENT,
        update={
            "messages": state["messages"] + [tool_message,insights_description_message]
        },
    )