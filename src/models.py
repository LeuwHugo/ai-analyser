from enum import Enum
from pydantic import BaseModel, Field, ValidationInfo, field_validator

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class Language(str, Enum):
    EN = "en"
    FR = "fr"
    ES = "es"
    DE = "de"
    OTHER = "other"


class ComplexityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TextAnalysis(BaseModel):
    """
    Structured output of the LLM analysis for a given input text.
    This model is the core contract between the LLM response and our app.
    """

    title: str = Field(
        ...,
        min_length=3,
        max_length=120,
        description="A short, descriptive title inferred from the text.",
    )
    summary: str = Field(
        ...,
        min_length=10,
        description="A 3-sentence summary of the text.",
    )
    key_points: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="The most important points extracted from the text.",
    )
    sentiment: Sentiment = Field(
        ...,
        description="Overall emotional tone of the text.",
    )
    language: Language = Field(
        ...,
        description="Detected language of the input text.",
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Relevant topic tags, lowercase, no spaces.",
    )
    complexity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reading complexity from 0.0 (simple) to 1.0 (very complex).",
    )
    complexity_level: ComplexityLevel = Field(
        ...,
        description="Human-readable complexity category derived from complexity_score.",
    )
    word_count: int = Field(
        ...,
        ge=0,
        description="Approximate word count of the input text.",
    )

    @field_validator("tags")
    @classmethod
    def tags_lowercase_no_spaces(cls, v: list[str]) -> list[str]:
        cleaned = []
        for tag in v:
            tag = tag.lower().strip().replace(" ", "-")
            if tag:
                cleaned.append(tag)
        return cleaned

    @field_validator("key_points")
    @classmethod
    def key_points_not_empty(cls, v: list[str]) -> list[str]:
        return [p.strip() for p in v if p.strip()]

    @field_validator("complexity_level")
    @classmethod
    def complexity_level_matches_score(
        cls, v: ComplexityLevel, info: ValidationInfo
    ) -> ComplexityLevel:
        score = info.data.get("complexity_score")
        if score is None:
            return v
        if score < 0.35:
            expected = ComplexityLevel.LOW
        elif score < 0.7:
            expected = ComplexityLevel.MEDIUM
        else:
            expected = ComplexityLevel.HIGH
        if v != expected:
            return expected
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "The Rise of Retrieval-Augmented Generation",
                    "summary": (
                        "RAG combines retrieval systems with generative models to produce "
                        "grounded, factual responses. It has become the dominant pattern "
                        "for enterprise AI applications. This article surveys key architectures "
                        "and evaluation benchmarks."
                    ),
                    "key_points": [
                        "RAG reduces hallucination by grounding answers in retrieved documents.",
                        "Hybrid search (dense + sparse) outperforms pure vector search.",
                        "RAGAS provides a standardised evaluation framework.",
                    ],
                    "sentiment": "positive",
                    "language": "en",
                    "tags": ["rag", "llm", "ai", "nlp"],
                    "complexity_score": 0.72,
                    "complexity_level": "high",
                    "word_count": 1240,
                }
            ]
        }
    }


class AnalysisRequest(BaseModel):
    """
    Input payload for the text analysis pipeline.
    Used by the CLI and (later) the FastAPI endpoint.
    """

    text: str = Field(
        ...,
        min_length=10,
        description="The raw text to analyse.",
    )
    provider: str = Field(
        default="openai",
        pattern="^(openai|anthropic)$",
        description="LLM provider to use: 'openai' or 'anthropic'.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response token by token.",
    )


class AnalysisResponse(BaseModel):
    """
    Full response envelope returned to the caller.
    Wraps TextAnalysis with metadata about the request.
    """

    analysis: TextAnalysis
    provider_used: str
    model_used: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)