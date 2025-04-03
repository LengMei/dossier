from typing import Annotated, Dict, Sequence, Optional, TypedDict, List, Any
import time
import re

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig 
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END

from jinja2 import Template

from agents.utils import wrap_agent_model
from agents.dossier_agent import Report
from tools.tools_search import search_internet, search_news
from agents.models import Reference, ReasoningNote, ResearchResult

# Define search tools
tools = [
    search_internet,
    search_news,
]

def extract_references(markdown_content: str) -> List[Reference]:
    """
    Extract references from markdown content
    
    Args:
        markdown_content: Markdown content with links
        
    Returns:
        List of Reference objects
    """
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', markdown_content)
    
    references = []
    for i, (text, url) in enumerate(links):
        references.append(Reference(
            id=f"ref_{i+1}",
            title=text,
            url=url,
            summary=None
        ))
    
    return references

def extract_reasoning(markdown_content: str) -> List[ReasoningNote]:
    """
    Extract reasoning sections from markdown content
    
    Args:
        markdown_content: Markdown content with reasoning sections
        
    Returns:
        List of ReasoningNote objects
    """
    reasoning_sections = []
    
    reasoning_headers = re.findall(
        r'## (Reasoning|Analysis|Methodology).*?\n(.*?)(?=\n##|\Z)',
        markdown_content, 
        re.DOTALL
    )
    
    for i, (header, content) in enumerate(reasoning_headers):
        reasoning_sections.append(ReasoningNote(
            id=f"reasoning_{i+1}",
            topic=header,
            content=content.strip(),
        ))
    
    return reasoning_sections

class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]   
    _internal_messages: Annotated[Sequence[BaseMessage], add_messages]
    company_name: Optional[str]
    research_topic: Optional[str]
    custom_topic: Optional[str]
    report: List[Report]
    agent_message: Optional[str]
    _search_query: Optional[str]
    search_results: Optional[List[Dict]]  # Store search results for citation
    _research_response: str
    research_result: Optional[Dict[str, Any]]  # Store research result for only current iteration

tools_node = ToolNode(tools, messages_key="_internal_messages")
today = time.strftime("%Y-%B-%d")

def create_research_subgraph(name, description, search_query_template, llm):
    """
    Creates a research agent subgraph specialized in a particular domain
    
    Args:
        name (str): Name of the research agent
        description (str): Description of the agent's specialty
        search_query_template (str): Template for search queries
        llm: Language model to use
        
    Returns:
        A compiled StateGraph that can be invoked
    """
    
    RESEARCH_INSTRUCTIONS = PromptTemplate.from_template(
    """You are the {name}, specialized in {description}.
    
    When a user asks about a company, conduct a comprehensive search to gather relevant and up-to-date information, including its background, financials, market position, leadership, latest news, and industry insights. Cross-reference multiple reliable sources to ensure accuracy and completeness. If internal knowledge is sufficient, provide a well-structured response directly; otherwise, supplement it with insights from online research. Always include reference links to your sources so users can verify and explore further. Present the information in a clear, concise, and business-professional manner, making sure to highlight key takeaways and actionable insights. Today is {today}.\\n
    
    Instructions:
    1. Conduct research based on the user's query
    2. Use the search tools to gather information. Use only relevant information from the search results for your research.
    3. For each important claim or insight, include:
        - Clear in-text citations (e.g., [1], [2]) with the corresponding link in markdown format
        - Your reasoning process for key conclusions
        - Proper attribution to sources
    4. Organize findings into a comprehensive markdown report with:
        - Executive summary
        - Key insights with reasoning
        - Detailed analysis with citations
        - IMPORTANT: Include a separate section titled "## Reasoning" or "## Analysis" or "## Methodology" that explains your analytical process
        - References section with numbered links
    5. Return a JSON with the structure: {{
        "company_name": "<company name>",
        "topic": "<research topic>", 
        "content": "<markdown content with citations>",
        "references": [
            {{"id": "1", "title": "Source title", "url": "https://source.url"}},
            ...
        ],
        "reasoning_notes": "<notes on key reasoning processes explaining analytical decisions>"
    }}
    
    Research request: {search_query}"""
    ).partial(name=name, description=description, today=today)
    
    ReportTemplate = """
    {% if report %}
    {\n\n Below is the previous report you can refer to. If the research request is not related to the previous report, just ignore it. If the research request is related to the previous report, use it to answer the research request as much as possible. If you find incorrect or incomplete information in the previous report, please correct it and highlight the corrections in your output. \n {report}}
    {% endif %}
    """
    
    async def format_query(state: ResearchState, config: RunnableConfig) -> ResearchState:
        """Generate the initial search query and research plan"""
        company_name = state.get("company_name")
        research_topic = state.get("research_topic")
        custom_topic = state.get("custom_topic")
        
        if research_topic != "other":
            
            if not company_name:
                error_message = AIMessage(content="I am specialized in conducting research on a company. Please specify which company you're interested in.")
                state["messages"] = state.get("messages", []) + [error_message]
                return state
        
            search_query = search_query_template.format(company_name=company_name)
        elif custom_topic:
            search_query = search_query_template.format(research_topic=custom_topic)
        else:
            error_message = AIMessage(content="Please specify the company name and/or the research topic you're interested in. Here are some examples: \n- Apple Inc. industry outlook \n- Microsoft financial status \n- Tesla vs. competitors \n- Google's latest news \n- How is the bonds market in China?")
            state["messages"] = state.get("messages", []) + [error_message]
            return state
            
        
        state["topic"] = research_topic if research_topic != "other" else custom_topic
        
        state["_search_query"] = search_query
        
        state["search_results"] = []
        
        state["_internal_messages"] = state["messages"].copy()
        return state

    async def research_with_tools(state: ResearchState, config: RunnableConfig) -> ResearchState:
        """Conduct research using tools"""
        report_content = ""
        for report in state["report"]:
            if report.company_name == state["company_name"]:
                report_content += report.report
            
        research_agent = wrap_agent_model(
            model=llm, 
            messages_key="_internal_messages",
            instructions=RESEARCH_INSTRUCTIONS.format(search_query=state["_search_query"]) + Template(ReportTemplate).render(report=report_content).strip(),
            tools=tools
        )
            
        response = await research_agent.ainvoke(state, config)
        
        state["_research_response"] = response.content
        state["_internal_messages"] = state["_internal_messages"] + [response]
        
        return state
        
    async def capture_search_results(state: ResearchState, config: RunnableConfig) -> ResearchState:
        """Capture search results for citation"""
        messages = state.get("_internal_messages", [])
        
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                search_result = {
                    "company_name": state.get("company_name", "Unknown Company"),
                    "topic": state.get("topic", "Unknown Topic"),
                    "tool": msg.tool_name if hasattr(msg, "tool_name") else "unknown",
                    "content": msg.content,
                    "timestamp": msg.additional_kwargs.get("timestamp", ""),
                    "metadata": msg.additional_kwargs.get("metadata", {})
                }
                
                state["search_results"] = state.get("search_results", []) + [search_result]
                break
                
        return state
        
    async def format_research(state: ResearchState, config: RunnableConfig) -> ResearchState:
        """Format research into structured output with citations and reasoning"""
        search_results = state.get("search_results", [])
        
        CITATION_INSTRUCTIONS = """
        Please organize your research into a high-quality report with:
        
        1. Make sure in-text citations are correctly included, especially for numbers and facts you cited from sources. Use the correct markdown format for in-text citation of the corresponding link, for example, [[url title]](https://link-url-here).
        2. A separate section titled "## Reasoning" that explains how the key conclusions are derived from the collected information.
        3. A "References" section at the end with numbered links to all referenced urls that are in the content.
        
        Here are the research results you can reference:\\n
        {research_results}
        """
        
        company_name = state.get("company_name", "Unknown company")
            
        if search_results:
            CITATION_INSTRUCTIONS += "\\n\\nHere are the search results you can reference:\\n"
            for i, result in enumerate(search_results, 1):
                title = result.get("metadata", {}).get("title", f"Search Result {i}")
                url = result.get("metadata", {}).get("url", "")
                source = result.get("metadata", {}).get("source", "")
                
                CITATION_INSTRUCTIONS += f"\\n[{i}] Title: {title}\\nURL: {url}\\nSource: {source}\\n"
        
        try:
            structured_agent = wrap_agent_model(
                model=llm, 
                messages_key="_internal_messages",
                instructions=CITATION_INSTRUCTIONS.format(research_results=state["_research_response"]),
                output_format=ResearchResult
            )
        
            structured_response = await structured_agent.ainvoke(state, config)
            
            research_result = {
                "company_name": structured_response.company_name or company_name,
                "topic": structured_response.topic,
                "content": structured_response.content,
                "references": structured_response.references,
                "reasoning_notes": structured_response.reasoning_notes,
            }
        except Exception as e:
            last_message = state["_internal_messages"][-1] if state.get("_internal_messages") else None
            content = last_message.content if hasattr(last_message, "content") else "No content available"
            
            topic = state.get("topic", name.replace("_subgraph", ""))
            
            references = []
            for i, result in enumerate(search_results, 1):
                title = result.get("metadata", {}).get("title", f"Search Result {i}")
                url = result.get("metadata", {}).get("url", "")
                references.append(Reference(id=str(i), title=title, url=url))
            
            research_result = {
                "company_name": company_name,
                "topic": topic,
                "content": f"## {topic.replace('_', ' ').title()}\\n\\n{content}",
                "references": references,
                "reasoning_notes": f"Unable to generate detailed reasoning notes. Error: {str(e)}",
            } 
            
        state["research_result"] = research_result
        
        return state

    def should_continue_tools(state: ResearchState) -> str:
        """Check if there are tool calls to process"""
        messages = state.get("_internal_messages", [])
            
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "process_tools"
        
        return "continue"

    workflow = StateGraph(ResearchState)
    
    workflow.add_node("format_query", format_query)
    workflow.add_node("research_with_tools", research_with_tools)
    workflow.add_node("process_tools", tools_node)
    workflow.add_node("capture_search_results", capture_search_results)
    workflow.add_node("format_research", format_research)
    
    workflow.add_edge("format_query", "research_with_tools")
    
    workflow.add_conditional_edges(
        "research_with_tools",
        should_continue_tools,
        {
            "process_tools": "process_tools",
            "continue": "format_research",
        }
    )
    
    workflow.add_edge("process_tools", "capture_search_results")
    workflow.add_edge("capture_search_results", "research_with_tools")
    workflow.add_edge("format_research", END)
    
    workflow.set_entry_point("format_query")
    
    return workflow.compile()
