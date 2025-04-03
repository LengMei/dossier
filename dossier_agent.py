import os
from typing import Dict, List, Any, TypedDict, Optional, Annotated, Sequence
from enum import Enum
import asyncio
import time

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from agents.utils import wrap_agent_model
from agents.chatbot import create_chatbot
from agents.models import (
    ResearchResult, 
    Report
)

from agents.research_agent import (
    industry_outlook_agent,
    company_financial_status_agent,
    company_peer_comparison_agent,
    general_research_agent,
    compile_raw_research_data,
)     
from agents.research_agent_subgraph import (
    extract_reasoning,
    extract_references,
)

from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from langchain_openai import ChatOpenAI

API_KEY = os.getenv("OPENAI_API_KEY")

llm_router = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY, temperature=0)
llm_writer = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY, temperature=0)

class ResearchTopic(str, Enum):
    INDUSTRY_OUTLOOK = "industry_outlook"
    COMPANY_FINANCIAL_STATUS = "company_financial_status"
    COMPANY_PEER_COMPARISON = "company_peer_comparison"
    OTHER = "other"

class MessageType(str, Enum):
    NEW_RESEARCH = "new_research"
    FOLLOW_UP = "follow_up"
    GENERAL_CHAT = "general_chat"

class NextAgent(str, Enum):
    CONVERSATION = "conversation_agent"
    INDUSTRY_OUTLOOK = "industry_outlook_agent"
    COMPANY_FINANCIAL = "company_financial_status_agent"
    COMPANY_PEER = "company_peer_comparison_agent"
    GENERAL_RESEARCH = "general_research_agent"
    REPORT_WRITER = "report_writing_agent"
    END = "end"

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    company_name: Optional[str]
    research_topic: Optional[ResearchTopic]
    custom_topic: Optional[str]
    message_type: Optional[MessageType]
    agent_message: Optional[str]
    next_agent: Optional[NextAgent]
    research: List[ResearchResult]
    report: List[Report]
    conversation_history: List[Dict[str, Any]]
    edited_reports: Optional[Dict[str, str]]  # Track edited reports for the UI

def get_initial_state() -> GraphState:
    return {
        "messages": [],
        "company_name": None,
        "research_topic": None,
        "custom_topic": None,
        "message_type": None,
        "next_agent": None,
        "agent_message": None,
        "research": [],
        "report": [],
        "conversation_history": [],
        "edited_reports": {},
    }

# ==== Supervisor Agent ====
SUPERVISOR_INSTRUCTIONS = """You are a supervisor for a corporate banking research system. Your job is to:
1. Determine the type of user request (new research, follow-up of an existing research topic on a given company, or general chat). New research requests are when the user wants to get a comprehensive research report on a company or on a specific research topic (e.g. industry outlook, financial status, peer comparison, financial news, etc.). Follow-up requests are when the user wants to continue a research that was already started and has some initial results from a previous research agent. General chat requests are when the user wants to chat about a general topic that can be answered by the conversation agent with its internal knowledge or with internet search.
    1.1 For new research requests and follow-up requests, extract the company name and determine the research topic
    1.2 For general chat requests, no company name is necessary, and the research topic is not applicable, just respond with a general chat response
2. If user requests a new research or a follow-up question on a previous research, once you have the company name and research topic, you should route the query to a research agent.
3. If you have all information you need for a confident routing decision, route the request to the appropriate agent. 
    2.1 Route to the conversation agent for general chat requests.
    2.2 Route to the research agent for new research requests and follow-up requests.
    2.3 If you are not sure about the user request, ask for clarification.

Research topics for new research requests available:
- industry_outlook: Analysis of the industry the company operates in and future outlook
- company_financial_status: Analysis of the company's financial health and performance
- company_peer_comparison: Comparison with similar companies in the same industry
- other: Any other research topic not covered above

After finalizing the routing decision, output a JSON with the following structure:
{
    "message_type": "new_research" | "follow_up" | "general_chat",
    "company_name": "Company Name" (if applicable),
    "research_topic": "industry_outlook" | "company_financial_status" | "company_peer_comparison" | "other" (if applicable),
    "custom_topic": "Brief description of custom research topic in at most five words" (if research_topic is "other"),
    "next_agent": "conversation_agent" | "industry_outlook_agent" | "company_financial_status_agent" | "company_peer_comparison_agent" | "general_research_agent" | "report_writing_agent",
    "agent_message": "Message to pass to the next agent"
}
"""

class SupervisorOutput(BaseModel):
    message_type: str = Field(description="Type of request: new_research, follow_up, or general_chat")
    company_name: Optional[str] = Field(description="Company name if applicable")
    research_topic: Optional[str] = Field(description="Research topic: industry_outlook, company_financial_status, company_peer_comparison, or other")
    custom_topic: Optional[str] = Field(description="Description of custom research topic if research_topic is other")
    next_agent: str = Field(description="Next agent to route to")
    agent_message: str = Field(description="Message to pass to the next agent")

async def supervisor_agent(state: GraphState, config: RunnableConfig) -> GraphState:
    full_state = state.copy()
    
    conversation_history = state.get("conversation_history", [])
    if conversation_history:
        history_messages = []
        for entry in conversation_history:
            if entry.get("role") == "supervisor":
                continue 
            if "content" in entry:
                history_messages.append(SystemMessage(content=f"Previous interaction - {entry['role']}: {entry['content']}"))
        
        if "messages" in full_state:
            full_state["messages"] = history_messages + full_state["messages"]
    
    supervisor_agent = wrap_agent_model(
        model=llm_router, 
        messages_key="messages",
        instructions=SUPERVISOR_INSTRUCTIONS,
        output_format=SupervisorOutput
    )
    
    response = await supervisor_agent.ainvoke(full_state, config)
    
    state["message_type"] = response.message_type
    state["company_name"] = response.company_name
    state["research_topic"] = response.research_topic
    state["custom_topic"] = response.custom_topic
    state["next_agent"] = response.next_agent
    state["agent_message"] = response.agent_message
    if (response.message_type in ["new_research", "follow_up"]):
        if (response.agent_message is not None):
            routing_message = AIMessage(content=response.agent_message)
            state["messages"].append(routing_message)
        if (response.company_name is None or response.research_topic is None):
            error_message = AIMessage(content="Please specify the company name and/or the research topic you're interested in. Here are some examples: \n- Apple Inc. industry outlook \n- Microsoft financial status \n- Tesla vs. competitors \n- Google's latest news \n- How is the bonds market in China?")
            state["messages"].append(error_message)
            state["next_agent"] = "__end__"
    
    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append({
        "role": "supervisor",
        "content": response.agent_message,
        "parsed": {
            "message_type": response.message_type,
            "company_name": response.company_name,
            "research_topic": response.research_topic,
            "custom_topic": response.custom_topic,
            "next_agent": response.next_agent,
            "agent_message": response.agent_message
        }
    })
        
    return state

# ==== Conversation Agent ====
CONVERSATION_INSTRUCTIONS = """You are a helpful corporate banking assistant. Answer general questions to the best of your ability.
Use the search tools when you need to look up specific information."""

async def conversation_agent(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Conversation agent that uses the chatbot subgraph but properly maintains conversation_history
    """
    chat_state = {
        "messages": state["messages"]
    }
    
    chatbot = create_chatbot(persist=False)
    response = await chatbot.ainvoke(chat_state, config)
    
    if response and "messages" in response:
        state["messages"].extend(response["messages"])
        
        if "conversation_history" not in state:
            state["conversation_history"] = []
            
        for message in response["messages"]:
            if not isinstance(message, ToolMessage):
                state["conversation_history"].append({
                    "role": "conversation_agent",
                    "content": message.content if hasattr(message, "content") else str(message)
                })
    
    return state

# ==== Report Writing Agent ====
REPORT_INSTRUCTIONS = PromptTemplate.from_template(
"""You are a report writing agent for corporate banking research. Your task is to write an comprehensive report on {number_of_topics} topics for {company_name} including {topics} using the research data provided.

Instructions:
1. Synthesize all the information into a coherent, professional report.
2. Make sure you include all topics, and each topic should be a separate section with potentially multiple subsections.
3. Format the report in clear sections with markdown formatting. 
4. Ensure each section has proper attribution for all claims and facts
5. Maintain all in-text citations with correct indexes and links from the original research, include as many citations as possible
6. Include all reasoning and analytical justifications in specific sections labeled with headings like "## Reasoning", "## Analysis" or "## Methodology"
7. Include a consolidated References section with all sources, which is already provided in the research data.

Structure your report with these parts:
- Executive Summary (highlight the key findings and recommendations)
- Research Analysis (detailed analysis of each topic)
    - Topic 1: Analysis
    - Topic 2: Analysis
    - Topic 3: Analysis
    - ...
- Reasoning and Methodology (separate section with detailed analysis of how conclusions were reached)
- Other relevant sections based on research
- References (numbered list with links)

Here are the research findings from multiple research agents for {company_name}:
{research_data}

Please synthesize this research data into a comprehensive report about {company_name} that maintains all citations and includes the reasoning behind key conclusions. 
The report should maintain academic rigor while being accessible to corporate banking professionals. Output the report in the field "report" in markdown format.

Important: Make sure to include specific sections with headings like "## Reasoning", "## Analysis", or "## Methodology" to explain the analytical process behind key findings. These sections will be displayed separately in the user interface.
""")

async def report_writing_agent(state: GraphState, config: RunnableConfig) -> GraphState:
    research_data = state.get("research", []) # company_name, topic, content, references, reasoning_notes
    company_name = state.get("company_name", "Unknown Company")
    
    if not research_data:
        error_message = AIMessage(content="I don't have any research data to generate a report. Please conduct research first. Start by providing the company name and research topic.")
        state["messages"].append(error_message)
        return state
    
    research_data = state.get("research", [])
    research_data = [r for r in research_data if r['company_name'] == company_name]
    research_data_str = await compile_raw_research_data(research_data)
    
    report_agent = wrap_agent_model(
        model=llm_writer, 
        messages_key="messages",
        instructions=REPORT_INSTRUCTIONS.format(research_data=research_data_str, company_name=company_name, number_of_topics=len(research_data), topics=", ".join([r['topic'] for r in research_data])),
        output_format=Report
    )
    
    try:
        response = await report_agent.ainvoke(state, config)
        
        references = extract_references(response.report)
        reasoning_sections = extract_reasoning(response.report)
        
        report = Report(
            company_name=response.company_name,
            report=response.report,
            materials=research_data,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            references=references,
            reasoning_sections=reasoning_sections
        )
        
        if "report" not in state:
            state["report"] = []
        state["report"].append(report)
        
        final_message = f"Corporate Banking Research Report for {company_name} has been generated successfully with citations and reasoning. You can now view the report in the right panel or update it with the edit button."
    except Exception as e:
        final_message = f"Error in generating report: {str(e)}. Please try again."
    
    state["messages"].append(AIMessage(content=final_message))
    
    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append({
        "role": "report_writing_agent",
        "content": "Report generated successfully with citations and reasoning"
    })
    
    if "edited_reports" not in state:
        state["edited_reports"] = {}
    if company_name not in state["edited_reports"]:
        state["edited_reports"][company_name] = final_message
    
    return state

# ==== Define Routing Logic ====
def router(state: GraphState) -> str:
    """Route to the next agent based on the state"""
    print("--------------------------------")
    print("Message: " + str(state['messages'][-1].content))
    print("Message Type: " + str(state['message_type']))
    print("Next Agent: " + str(state['next_agent']))
    
    if "next_agent" not in state or state["next_agent"] is None:
        return "supervisor_agent"
        
    if state["next_agent"] == NextAgent.CONVERSATION:
        return "conversation_agent"
    elif state["next_agent"] == NextAgent.INDUSTRY_OUTLOOK:
        return "industry_outlook_agent"
    elif state["next_agent"] == NextAgent.COMPANY_FINANCIAL:
        return "company_financial_status_agent"
    elif state["next_agent"] == NextAgent.COMPANY_PEER:
        return "company_peer_comparison_agent"
    elif state["next_agent"] == NextAgent.GENERAL_RESEARCH:
        return "general_research_agent"
    elif state["next_agent"] == NextAgent.REPORT_WRITER:
        return "report_writing_agent"
    else:
        return "__end__"

# ==== Build Main Graph ====
def build_graph(persist=True):
    """
    Build the main research agent graph with proper conversation history tracking
    """
    workflow = StateGraph(GraphState)
    
    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("conversation_agent", conversation_agent)
    workflow.add_node("industry_outlook_agent", industry_outlook_agent)
    workflow.add_node("company_financial_status_agent", company_financial_status_agent)
    workflow.add_node("company_peer_comparison_agent", company_peer_comparison_agent)
    workflow.add_node("general_research_agent", general_research_agent)
    workflow.add_node("report_writing_agent", report_writing_agent)
    
    workflow.add_conditional_edges(
        "supervisor_agent",
        router,
        {
            "supervisor_agent": "supervisor_agent",
            "conversation_agent": "conversation_agent",
            "industry_outlook_agent": "industry_outlook_agent",
            "company_financial_status_agent": "company_financial_status_agent",
            "company_peer_comparison_agent": "company_peer_comparison_agent",
            "general_research_agent": "general_research_agent",
            "report_writing_agent": "report_writing_agent",
            "__end__": END,
        }
    )

    for agent_name in [
        "industry_outlook_agent", 
        "company_financial_status_agent", 
        "company_peer_comparison_agent", 
        "general_research_agent",
    ]:
        workflow.add_edge(agent_name, "report_writing_agent")
    
    workflow.add_edge("report_writing_agent", END)
    workflow.add_edge("conversation_agent", END)
    
    workflow.set_entry_point("supervisor_agent")

    if persist:
        agent = workflow.compile(
            checkpointer=MemorySaver(),
        )
    else:
        agent = workflow.compile()
    return agent

agent = build_graph()