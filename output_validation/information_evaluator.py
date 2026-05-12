from pydantic import BaseModel, Field


class InformationEvaluation(BaseModel):
    """Evidence sufficiency judgment produced by the information evaluator node."""

    information_complete: bool = Field(
        description="True when the messages contain enough retrieved or prior evidence to answer.",
        examples=[False],
    )
    information_complete_explanation: str = Field(
        description="A concise explanation of why the available information is or is not sufficient.",
        examples=["The retrieved passages mention pricing but do not cover refund windows."],
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Specific evidence still needed when information_complete is false.",
        examples=[["Enterprise tier refund window", "Consumer tier refund window"]],
    )
