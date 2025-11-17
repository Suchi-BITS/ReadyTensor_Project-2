import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:    
    sys.path.insert(0, root_dir)
from langchain_core.messages import AIMessage, ToolCall  
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from agentic_tools import AgentScratchpadTools
from langchain_openai import ChatOpenAI
from utils.prompt_loader import load_prompt_from_hub
from utils.logger_setup import setup_execution_logger
from schema.state import SupervisorState
from langchain_core.messages import SystemMessage
logger = setup_execution_logger()
AGENT_ID = "insights_agent"
STATE_SCRATCHPAD_KEY = "insights_agent_scratchpad"
from agent_tools import inject_restricted_pandas

class InsightsOutput(BaseModel):
    summary: str
    details: str
    new_insights_filename: str
    execution_status: str

#===============OpenAI model==========================
llm = ChatOpenAI(
    model="gpt-4.1",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
#=======================================================

def call_insights_agent(state : SupervisorState):   
    # Invoke load_scratchpad_from_cel tool and write its output into the scratchpad
    wrapped_repl = inject_restricted_pandas(state)
    prompt = load_prompt_from_hub("CEL_insights_agent")
    tool_factory = AgentScratchpadTools(
        agent_id=AGENT_ID,
        state_scratchpad_key=STATE_SCRATCHPAD_KEY
    )
    load_scratchpad_from_cel = tool_factory.get_load_from_cel_tool()
    save_tool = tool_factory.get_save_to_cel_tool()    
    write_tool = tool_factory.get_write_to_memory_tool()  
    read_tool = tool_factory.get_read_from_memory_tool()

    tools = [
        wrapped_repl,   
        read_tool,
        write_tool
]
    print("INSIDE INSIGHTS AGENT")
    
    system_message = SystemMessage(  
        content=prompt
    )  
    agent = create_react_agent(
    llm,
    tools,
    prompt = system_message,
    state_schema=SupervisorState,
    response_format=InsightsOutput
    # debug = True
)
    tool_call = ToolCall(
        name="load_scratchpad_from_cel",
        args={},
        id="load_scratchpad_from_cel_call"
    )

    tool_node = ToolNode([load_scratchpad_from_cel])  

    # Execute with proper state context  
    tool_result = tool_node.invoke({  
        "messages": [AIMessage(content="", tool_calls=[tool_call])],  
        "session_id": state["session_id"],  
    })  
      
    # Extract the response  
    load_cel_response = tool_result["messages"][-1].content  
    state['insights_agent_scratchpad'] += f"\n[Previous Context LOADED]\n{load_cel_response if load_cel_response else 'No previous context found.Treat this as first query in the session and answer independently.'}\n"  

    logger.info(f'\n[BEFORE_INSIGHTS_EXECUTION]{state["insights_agent_scratchpad"]}\n')
    response = agent.invoke(state, config={"recursion_limit": 50})
    state['insights_summary'] = response.get('structured_response').summary +"\nOther notes and details : " + response.get('structured_response').details
    dataframe_path = os.path.join(state['output_dir'], response.get('structured_response').new_insights_filename)
    logger.info(f'\nINSIGHTS AGENT CSV : {dataframe_path}\n')
    state['dataframe_path'] = dataframe_path
    scratchpad_content = response.get("insights_agent_scratchpad", "") 
    
    state['insights_agent_scratchpad'] = scratchpad_content
    if scratchpad_content.strip():  
        try:  
            save_result = save_tool.invoke({  
                "scratchpad": scratchpad_content,  
                "state": state  
            })  
            print(f"Saved scratchpad to CEL: {save_result}")  
        except RuntimeError as e:  
            if "session_id must be provided" in str(e):  
                print(f"Warning: Cannot save to CEL - {e}")  
            else:  
                print(f"Error saving scratchpad to CEL: {e}")  
        except Exception as e:  
            print(f"Unexpected error saving scratchpad to CEL: {e}")  
    return {
        "messages" : state['messages'] + [AIMessage(content=f"Insights generation has been completed.Move to next plan of action. Scratchpad thoughts : {state['insights_agent_scratchpad']}")],
        'insights_agent_scratchpad'  :scratchpad_content,
        'dataframe_path' : dataframe_path,
        'insights_summary' : state['insights_summary']
    }
