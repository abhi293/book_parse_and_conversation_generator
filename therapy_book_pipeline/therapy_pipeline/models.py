from pydantic import BaseModel, Field
from typing import Literal

Modality = Literal["cbt_beck", "person_centered_rogers", "cross_modality", "uncategorized"]

class Strategy(BaseModel):
    strategy_id: str = ""
    strategy_name: str
    modality: Modality
    language: str = "und"
    description: str
    when_to_use: str
    how_to_apply: str
    example_phrasing: str = Field(description="A short newly written illustrative phrase; do not reproduce book text.")
    limitations_or_cautions: str = ""
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0, le=1)

class ScenarioSeed(BaseModel):
    seed_id: str = ""
    title: str
    situation_summary: str
    presenting_problem: str
    typical_thoughts_or_feelings: list[str] = Field(default_factory=list)
    observable_behaviors: list[str] = Field(default_factory=list)
    initial_anxiety_estimate: int = Field(ge=1, le=10)
    modality_relevance: list[Modality] = Field(default_factory=list)
    language: str = "und"
    source_refs: list[str] = Field(default_factory=list)
    is_book_illustration: bool = False
    confidence: float = Field(default=0.7, ge=0, le=1)

class ChunkExtraction(BaseModel):
    strategies: list[Strategy] = Field(default_factory=list)
    scenario_seeds: list[ScenarioSeed] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
