from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
import re
import os

from agents.research_agent_subgraph import (
    create_research_subgraph, 
    Reference, 
    ReasoningNote,
)

from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from langchain_openai import ChatOpenAI
llm_research = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

industry_outlook_subgraph = create_research_subgraph(
    name="industry_outlook_subgraph",
    description="Research agent specialized in analyzing industry outlook and market trends",
    search_query_template="""
    Research the industry outlook and market trends for {company_name}. Focus on:
    1. Current market position and competitive landscape
    2. Industry growth trends and forecasts
    3. Key market drivers and challenges
    4. Regulatory environment and compliance
    5. Technological advancements and innovation
    6. Market opportunities and threats
    7. Industry-specific metrics and KPIs
    8. Future outlook and predictions
    
    Include specific data points, statistics, and expert analysis where available.
    """,
    llm=llm_research
)


company_financial_status_subgraph = create_research_subgraph(
    "company_financial_status_subgraph",
    "analyzing company financial health with precise citations to financial reports and reasoned analysis",
    "What is the current financial status of {company_name}? Include revenue, profit margins, debt levels, cash flow, recent financial performance, and key financial ratios. Cite specific quarterly reports, annual reports, and financial analyst assessments. Include a dedicated '## Analysis' section explaining your analytical process.",
    llm_research
)

company_peer_comparison_subgraph = create_research_subgraph(
    "company_peer_comparison_subgraph",
    "comparing companies with peers using data-driven benchmarking and comparative analysis with citations",
    "How does {company_name} compare to its industry peers? Include market position, competitive advantages, financial performance relative to competitors, market share data, and industry rankings. Use specific metrics and benchmarks with proper citations. Include a dedicated '## Methodology' section explaining your analytical process.",
    llm_research
)

general_research_subgraph = create_research_subgraph(
    "general_research_subgraph",
    "conducting comprehensive business research with citations and reasoned analysis",
    "{research_topic}? Include factual information from reliable sources with proper citations and reasoned analysis of key business decisions. Include a dedicated '## Reasoning' section explaining your analytical process.",
    llm_research
)

def merge_references(references_list: List[List[Reference]]) -> List[Reference]:
    """Merge references from multiple research results with proper renumbering"""
    all_references = []
    url_to_id = {} 
    
    for references in references_list:
        if not references:
            continue
            
        for ref in references:
            if isinstance(ref, dict):
                url = ref.get("url", "")
                title = ref.get("title", "Untitled Source")
            else:
                url = ref.url if hasattr(ref, "url") else ""
                title = ref.title if hasattr(ref, "title") else "Untitled Source"
                
            if url and (url not in url_to_id):
                new_id = str(len(all_references) + 1)
                
                new_ref = Reference(
                    id=new_id,
                    title=title,
                    url=url,
                    summary=None
                )
                
                url_to_id[url] = new_id
                all_references.append(new_ref)
    
    return all_references

def update_reference_index(content: str, references: List[Reference]) -> str:
    """
    Update reference indices in content to match the merged reference list
    
    Args:
        content: The content with reference citations
        references: List of merged references with new indices
        
    Returns:
        Updated content with new reference indices
    """
    url_to_new_index = {ref.url: ref.id for ref in references}
    
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    links = re.findall(link_pattern, content)
    
    old_to_new_index = {}
    for text, url in links:
        if url in url_to_new_index:
            old_index_match = re.search(r'\[?(\d+)\]?', text)
            if old_index_match:
                old_index = old_index_match.group(1)
                old_to_new_index[old_index] = url_to_new_index[url]
    
    sorted_old_indices = sorted(old_to_new_index.keys(), key=len, reverse=True)
    
    updated_content = content
    for old_index in sorted_old_indices:
        new_index = old_to_new_index[old_index]
        patterns = [
            rf'\[{old_index}\]',  # [1]
            rf'\[ref_{old_index}\]',  # [ref_1]
            rf'\[{old_index}\]',  # [1] (with any prefix)
        ]
        
        for pattern in patterns:
            updated_content = re.sub(pattern, f'[{new_index}]', updated_content)
    
    return updated_content

def merge_reasoning_notes(reasoning_sections_list: List[tuple[str, str]]) -> List[ReasoningNote]:
    """Merge reasoning sections from multiple research results"""
    all_reasoning = []
    topic_to_id = {} 
    
    for section in reasoning_sections_list:
        new_id = f"reasoning_{len(all_reasoning) + 1}"
            
        new_note = ReasoningNote(
            id=new_id,
            topic=section[0],
            content=section[1] or "" 
        )
        
        topic_to_id[section[0]] = new_id
        all_reasoning.append(new_note)
    
    return all_reasoning

async def compile_raw_research_data(research_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compile all research results into a final report with consolidated references and reasoning"""
        
    references_list = []
    reasoning_notes_list = []
    
    for result in research_data:
        if "references" in result:
            references_list.append(result["references"])
    merged_references = merge_references(references_list)
    
    sections = []
    
    for result in research_data:
        topic = result.get("topic", "Unknown Topic")
        content = result.get("content", "")
        
        updated_content = update_reference_index(content, merged_references)
        
        sections.append(f"## {topic}")
        sections.append(updated_content)
        sections.append("\n")
    
        if "reasoning_notes" in result:
            reasoning_notes_list.append((topic, result["reasoning_notes"]))
    merged_reasoning = merge_reasoning_notes(reasoning_notes_list)
    
    references_section = ["## References"]
    for ref in merged_references:
        references_section.append(f"[{ref.id}] {ref.title}: {ref.url}")
    
    reasoning_section = ["## Reasoning and Methodology"]
    for note in merged_reasoning:
        reasoning_section.append(f"### {note.topic}")
        reasoning_section.append(note.content)
    
    final_content = "\n".join([
        "# Material for Research Report Writing",
        "\n".join(sections),
        "\n".join(reasoning_section) if merged_reasoning else "",
        "\n".join(references_section)
    ])
    return final_content

async def industry_outlook_agent(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """Wrapper for the industry outlook research subgraph"""
    subgraph_state = {
        "messages": state["messages"],
        "company_name": state["company_name"],
        "research_topic": state["research_topic"],
        "custom_topic": state["custom_topic"],
        "report": state["report"],
    }
    
    result = await industry_outlook_subgraph.ainvoke(subgraph_state, config)
    
    if result and ("research_result" in result):
        if "research" not in state:
            state["research"] = []
        
        research_data = result["research_result"]
        
        enhanced_research = {
            "company_name": research_data.get("company_name", state.get("company_name", "Unknown Company")),
            "topic": research_data.get("topic", "industry_outlook"),
            "content": research_data.get("content", ""),
            "references": research_data.get("references", []), 
            "reasoning_notes": research_data.get("reasoning_notes", ""),
        }
        
        state["research"].append(enhanced_research)
        
        if "messages" in result:
            state["messages"].extend(result["messages"])
    
    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append({
        "role": "industry_outlook_agent",
        "content": "Industry outlook research completed with citations and reasoning"
    })
    
    return state

async def general_research_agent(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """Wrapper for the general research subgraph"""
    subgraph_state = {
        "messages": state["messages"],
        "company_name": state["company_name"],
        "research_topic": state["research_topic"],
        "custom_topic": state["custom_topic"],
        "report": state["report"],
    }
    
    result = await general_research_subgraph.ainvoke(subgraph_state, config)
    
    if result and "research_result" in result:
        if "research" not in state:
            state["research"] = []
            
        research_data = result["research_result"]
        
        enhanced_research = {
            "company_name": research_data.get("company_name", state.get("company_name", "Unknown Company")),
            "topic": research_data.get("topic", state.get("custom_topic", "general_research")),
            "content": research_data.get("content", ""),
            "references": research_data.get("references", []),
            "reasoning_notes": research_data.get("reasoning_notes", ""),
        }
            
        state["research"].append(enhanced_research)
        
        if "messages" in result:
            state["messages"].extend(result["messages"])
    
    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append({
        "role": "general_research_agent",
        "content": "General research completed with citations and reasoning"
    })
    
    return state

async def company_peer_comparison_agent(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """Wrapper for the company peer comparison research subgraph"""
    subgraph_state = {
        "messages": state["messages"],
        "company_name": state["company_name"],
        "research_topic": state["research_topic"],
        "custom_topic": state["custom_topic"],
        "report": state["report"],
    }
    
    result = await company_peer_comparison_subgraph.ainvoke(subgraph_state, config)
    
    if result and "research_result" in result:
        if "research" not in state:
            state["research"] = []
            
        research_data = result["research_result"]
        
        enhanced_research = {
            "company_name": research_data.get("company_name", state.get("company_name", "Unknown Company")),
            "topic": research_data.get("topic", "company_peer_comparison"),
            "content": research_data.get("content", ""),
            "references": research_data.get("references", []),
            "reasoning_notes": research_data.get("reasoning_notes", ""),
        }
            
        state["research"].append(enhanced_research)
        
        if "messages" in result:
            state["messages"].extend(result["messages"])
    
    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append({
        "role": "company_peer_comparison_agent",
        "content": "Company peer comparison research completed with citations and reasoning"
    })
    
    return state

async def company_financial_status_agent(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """Wrapper for the company financial status research subgraph"""
    subgraph_state = {
        "messages": state["messages"],
        "company_name": state["company_name"],
        "research_topic": state["research_topic"],
        "custom_topic": state["custom_topic"],
        "report": state["report"],
    }
    
    result = await company_financial_status_subgraph.ainvoke(subgraph_state, config)
    
    if result and "research_result" in result:
        if "research" not in state:
            state["research"] = []
            
        research_data = result["research_result"]
        
        enhanced_research = {
            "company_name": research_data.get("company_name", state.get("company_name", "Unknown Company")),
            "topic": research_data.get("topic", "company_financial_status"),
            "content": research_data.get("content", ""),
            "references": research_data.get("references", []),
            "reasoning_notes": research_data.get("reasoning_notes", ""),
        }
            
        state["research"].append(enhanced_research)
        
        if "messages" in result:
            state["messages"].extend(result["messages"])
    
    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append({
        "role": "company_financial_status_agent",
        "content": "Company financial status research completed with citations and reasoning"
    })
    
    return state