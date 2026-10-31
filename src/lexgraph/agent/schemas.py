"""Structured outputs requested from the language models.

Every field is required (nullable where a value may be absent) and has no default: Groq's
strict JSON-schema mode rejects optional fields and defaults.
"""

from typing import Literal

from pydantic import BaseModel, Field

LawCode = Literal["CLT", "CDC", "LGPD", "MCI", "CF", "ADCT"]


class ArticleRef(BaseModel):
    """An explicit reference to an article found in the question."""

    law: LawCode | None = Field(description="Law of the article, or null when not stated.")
    article: str = Field(description='Article number without ordinal sign, e.g. "7" or "59-A".')


class QueryAnalysis(BaseModel):
    """How the agent should handle the latest user message."""

    intent: Literal["legal_question", "smalltalk", "out_of_scope"] = Field(
        description=(
            "legal_question: any question about rights, duties or Brazilian law; "
            "smalltalk: greetings or questions about the assistant itself; "
            "out_of_scope: clearly unrelated to law (recipes, sports, programming...)."
        )
    )
    standalone_question: str = Field(
        description="The question rewritten to be understandable without the conversation."
    )
    search_query: str = Field(
        description="Search query in Portuguese with the legal terms likely used in the statute."
    )
    sub_queries: list[str] = Field(
        description=(
            "Up to 2 extra search queries for other legal points of the question (an exception, "
            "a related rule, the other law of a cross-law question); empty for a single point."
        )
    )
    laws: list[LawCode] = Field(description="Laws the question clearly refers to; may be empty.")
    articles: list[ArticleRef] = Field(description="Articles explicitly cited; may be empty.")


class RelevanceGrade(BaseModel):
    """Which retrieved passages help answer the question."""

    reasoning: str = Field(description="One or two sentences explaining the decision.")
    relevant: list[int] = Field(description="Numbers [n] of the passages that are relevant.")


class QueryRewrite(BaseModel):
    """A better search query after a search that found nothing useful."""

    search_query: str = Field(description="New search query using the vocabulary of the statute.")
    laws: list[LawCode] = Field(description="Laws to search in; empty means all laws.")


class GroundingCheck(BaseModel):
    """Whether every legal claim in the answer is supported by its sources."""

    unsupported_claims: list[str] = Field(
        description="Claims about the law that the sources do not support; empty if none."
    )
    grounded: bool = Field(description="True when unsupported_claims is empty.")
