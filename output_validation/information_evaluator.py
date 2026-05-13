from pydantic import BaseModel, Field


class InformationEvaluation(BaseModel):
    """Evidence sufficiency judgment produced by the information evaluator node."""

    is_information_complete: bool = Field(
        description=(
            "True when retrieved passages fully support a grounded answer for the user need "
            "implied by the parsed queries."
        ),
        examples=[False],
    )
    missing_evidence_details: list[str] = Field(
        default_factory=list,
        description=(
            "When is_information_complete is false: required, non-empty list of detailed analyses. "
            "Each item should clearly describe (1) what the current retrieved passages do cover or "
            "partially address, and (2) what substantive evidence, facts, or scope is still absent. "
            "When is_information_complete is true: use an empty list."
        ),
        examples=[
            [
                "Retrieved passages describe general refund policy and timelines but do not name "
                "Enterprise vs Consumer tier-specific windows; need explicit per-tier deadlines.",
            ]
        ],
    )
