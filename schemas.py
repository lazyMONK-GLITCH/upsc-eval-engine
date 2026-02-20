from pydantic import BaseModel, Field
from typing import Optional, List

class DocumentNode(BaseModel):
    doc_id: str = Field(..., description="Unique identifier for the source material")
    title: str = Field(..., description="Title of the UPSC document/book")

class TopicNode(BaseModel):
    name: str = Field(..., description="Name of the syllabus topic")
    paper: str = Field(..., description="UPSC Paper (e.g., GS-2)")

class ChunkNode(BaseModel):
    chunk_id: str = Field(..., description="Unique ID for the text chunk")
    text: str = Field(..., description="The raw text extracted from the document")
    vector_embedding: Optional[List[float]] = Field(None, description="Gemini Flash 1.5 generated embedding")

class EntityNode(BaseModel):
    name: str = Field(..., description="Specific named entity (e.g., Article 14, Supreme Court)")
    entity_type: str = Field(..., description="Category of the entity")

class PYQNode(BaseModel):
    pyq_id: str = Field(..., description="Unique ID for the past year question")
    year: int = Field(..., description="Year the question was asked")
    text: str = Field(..., description="The exact question text")

class CurrentAffairNode(BaseModel):
    headline: str = Field(..., description="Headline of the news event")
    date: str = Field(..., description="Date of occurrence (YYYY-MM-DD)")
# --- INFERENCE PIPELINE SCHEMAS ---

class RouteQuery(BaseModel):
    """Structured output for the Groq intent router."""
    intent: str = Field(..., description="Must be exactly one of: 'concept_explanation', 'pyq_analysis', or 'general_search'")
    entities: List[str] = Field(..., description="List of specific UPSC entities (Articles, Judgments, terms) extracted from the query for Neo4j lookup")