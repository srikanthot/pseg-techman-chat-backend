"""
Pydantic models for API request and response validation.

ChatResponse is the stable contract consumed by the Power Apps / PCF integration.
Do not change field names without coordinating with the consuming team.
"""

from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request body for both /chat and /chat/stream."""

    question: str
    session_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "How do I perform a pressure test on a gas service line?",
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                }
            ]
        }
    }


class Citation(BaseModel):
    """A single source reference from the retrieved manuals."""

    source: str
    title: str = ""
    section: str = ""
    page: str = ""
    url: str = ""
    chunk_id: str = ""


class ChatResponse(BaseModel):
    """Response body from the non-streaming POST /chat endpoint.

    This is the primary integration contract for Power Apps / PCF consumers.
    """

    answer: str
    citations: list[Citation]
    session_id: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "According to the manual [1], the pressure test procedure...\n\nSources:\n- gas_service_manual.pdf, Section: Pressure Testing",
                    "citations": [
                        {
                            "source": "gas_service_manual.pdf",
                            "title": "Gas Service Installation Manual",
                            "section": "Field Procedures > Pressure Testing",
                            "page": "",
                            "url": "https://storage.blob.core.windows.net/manuals/gas_service_manual.pdf",
                            "chunk_id": "gas_service_manual.pdf_chunk_042",
                        }
                    ],
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                }
            ]
        }
    }


class CitationsPayload(BaseModel):
    """Payload for the SSE 'citations' named event in /chat/stream."""

    citations: list[Citation]
