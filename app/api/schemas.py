"""
Pydantic request / response models for the chat API.

ChatResponse is the stable contract consumed by Power Apps / PCF integrations.
Do not rename fields without coordinating with the consuming team.
"""

from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request body for both POST /chat and POST /chat/stream."""

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
    """A single source reference from the retrieved technical manuals."""

    source: str
    title: str = ""
    section: str = ""
    page: str = ""
    url: str = ""
    chunk_id: str = ""


class ChatResponse(BaseModel):
    """Response body from the non-streaming POST /chat endpoint.

    Primary integration contract for Power Apps / PCF consumers.
    """

    answer: str
    citations: list[Citation]
    session_id: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "According to the manual [1], the pressure test procedure...\n\nSources:\n- gas_service_manual.pdf",
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
    """Payload for the SSE ``citations`` named event in POST /chat/stream."""

    citations: list[Citation]
