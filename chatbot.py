import os

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import time

from agents.utils import wrap_agent_model
from tools.tools_search import search_internet, search_news

from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

API_KEY = os.getenv("OPENAI_API_KEY")

llm_chatbot = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY, temperature=0)

today = time.strftime("%Y-%B-%d")

class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

CHAT_AGENT_SYSTEM_INSTRUCTION = """ 
You are an advanced AI assistant with both extensive internal knowledge and real-time search capabilities. When answering user queries, always first draw upon your built-in knowledge to provide well-structured, insightful, and relevant responses. If additional or up-to-date information is required, perform an online search to supplement your answer with the latest credible details. When incorporating online information, ensure accuracy and seamlessly integrate it into your response. Always include reference links to the sources used so users can verify and explore further. Your goal is to provide the best possible answer by combining internal expertise with external knowledge (when necessary) in a clear, concise, and trustworthy manner. Today is {today}."
"""

def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

tools = [
    search_internet,
    search_news
]

node__tools = ToolNode(tools)

async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    model_runnable = wrap_agent_model(llm_chatbot, messages_key="messages", instructions=CHAT_AGENT_SYSTEM_INSTRUCTION.format(today=today), tools=tools)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"
    
def create_chatbot(persist=True):
    # Define the graph
    graph = StateGraph(AgentState)
    
    graph.add_node("agent", acall_model)
    
    # Add the tools node
    graph.add_node("tools", node__tools)
    
    # Add conditional edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    
    # Add edge from tools back to agent
    graph.add_edge("tools", "agent")
    
    # Set the entry point
    graph.set_entry_point("agent")

    if persist:
        chatbot = graph.compile(
            checkpointer=MemorySaver(),
        )
    else:
        chatbot = graph.compile()
    return chatbot

chatbot = create_chatbot()
