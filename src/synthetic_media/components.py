"""Stable contracts for independently implemented model adapters."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

STAGES = ("face_processing", "face_swapping", "video_consistency", "voice_cloning",
          "lip_synchronization", "facial_reenactment", "multimodal_integration", "quality_optimization")

@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    metadata: dict = field(default_factory=dict)

@dataclass(frozen=True)
class Context:
    job_id: str
    output_dir: Path
    device: str = "cpu"
    allow_low_detail: bool = False
    license_note: str = ""
    instruction: str = ""
    edit_scope: str = "face"

class Component(Protocol):
    name: str
    def process(self, inputs: dict[str, Artifact], context: Context) -> dict[str, Artifact]: ...

class Registry:
    def __init__(self):
        self._components: dict[str, Component] = {}

    def register(self, component: Component):
        if component.name not in STAGES:
            raise ValueError(f"Unknown stage: {component.name}")
        if component.name in self._components:
            raise ValueError(f"Already registered: {component.name}")
        self._components[component.name] = component

    def require(self, name: str) -> Component:
        if name not in self._components:
            raise ValueError(f"Stage '{name}' has no installed model adapter")
        return self._components[name]

    def capabilities(self):
        return {name: {"available": name in self._components} for name in STAGES}
