from typing import List, Optional
from pydantic import BaseModel, Field

class Reference(BaseModel):
    id: str = Field(description="Reference ID (e.g., '1', '2')")
    title: str = Field(description="Reference title")
    url: str = Field(description="Reference URL")
    summary: Optional[str] = Field(None, description="Optional summary of the reference")

class ReasoningNote(BaseModel):
    id: str = Field(description="Reasoning identifier")
    topic: str = Field(description="Topic this reasoning relates to")
    content: str = Field(description="Detailed reasoning explanation")
    
class ResearchResult(BaseModel):
    company_name: Optional[str] = Field(None, description="Company name")
    topic: str = Field(description="Research topic")
    content: str = Field(description="The comprehensive research content in markdown format with in-text citations")
    references: List[Reference] = Field(default_factory=list, description="List of reference objects with id, title, url, and optional summary")
    reasoning_notes: Optional[str] = Field(None, description="Notes explaining reasoning process for key claims")
    
class ReportTopicOutput(BaseModel):
    company_name: Optional[str] = Field(description="Company name")
    topic: str = Field(description="Topic title")
    content: str = Field(description="Content of the topic derived from the relevant research results in markdown format")
    
class Report(BaseModel):
    company_name: Optional[str] = Field(description="Company name")
    report: str = Field(description="final report in markdown format")
    materials: List[ReportTopicOutput] = Field(default_factory=list, description="List of materials used to generate the report")
    timestamp: Optional[str] = Field(None, description="Report generation timestamp for UI display")
    references: List[Reference] = Field(default_factory=list, description="List of reference objects with id, title, url, and optional summary")
    reasoning_sections: List[ReasoningNote] = Field(default_factory=list, description="List of reasoning notes with id, topic, and content")