from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class ProgressEvent:
    task: str
    phase: str
    message: str
    current: int
    total: int

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


ProgressEmitter = Callable[[ProgressEvent], None]
