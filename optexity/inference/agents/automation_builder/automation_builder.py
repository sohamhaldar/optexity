import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from optexity.inference.agents.automation_builder.prompt import system_prompt
from optexity.inference.models.llm_model import LLMModel
from optexity.schema.token_usage import TokenUsage


class BuiltStep(BaseModel):
    action: Literal["input", "click", "select_dropdown", "go_back", "upload_file"]
    command: str = Field(description="Copied verbatim from the recording.")
    prompt_instructions: str = Field(description="How a person would name this element.")
    input_text: str | None = Field(default=None, description="Text to type, for input steps.")
    press_enter: bool = Field(default=False)


class BuiltAutomation(BaseModel):
    steps: list[BuiltStep]


class AutomationBuilderAgent:
    """Chooses which recorded actions a deterministic automation should replay."""

    def __init__(self, model: LLMModel):
        self.model = model

    def build(
        self, recording: list[dict[str, Any]], locator_docs: str
    ) -> tuple[BuiltAutomation, TokenUsage]:
        """Steps to replay, drawn only from `recording`.

        Callers must reject any command absent from the recording: the model can
        return a plausible locator nobody verified.
        """
        prompt = f"""
        [LOCATOR REFERENCE]
        {locator_docs}
        [/LOCATOR REFERENCE]

        [RECORDING]
        {json.dumps(recording, indent=2)}
        [/RECORDING]
        """
        response, token_usage = self.model.get_model_response_with_structured_output(
            prompt=prompt,
            response_schema=BuiltAutomation,
            system_instruction=system_prompt,
        )
        return response, token_usage
