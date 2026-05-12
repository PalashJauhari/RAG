from pydantic import BaseModel, Field


class OrchestratorOutput(BaseModel):
    """Routing decision produced by the graph orchestrator node."""

    query: str = Field(
        description=(
            "The single rewritten retrieval query. Leave empty only when "
            "clarification_required is true."
        ),
        examples=["enterprise refund policy cancellation window"],
    )
    query_decomposition: bool = Field(
        description="Whether the query contains multiple parts that should be split before retrieval.",
        examples=[False],
    )
    query_expansion: bool = Field(
        description="Whether each retrieval query should be expanded with related terminology.",
        examples=[True],
    )
    retrieval_required: bool = Field(
        description=(
            "True when the current message context does not already contain enough evidence "
            "to answer the user."
        ),
        examples=[True],
    )
    clarification_required: bool = Field(
        default=False,
        description="True only when the user request is too ambiguous to search safely.",
        examples=[False],
    )
    clarification_question: str = Field(
        default="",
        description="A concise question for the user when clarification_required is true.",
        examples=["Are you asking about the enterprise refund policy or the consumer refund policy?"],
    )
    routing_explanation: str = Field(
        description="A brief observable explanation of the routing decision, not hidden reasoning.",
        examples=["The query is a comparison, so it should be decomposed before retrieval."],
    )
