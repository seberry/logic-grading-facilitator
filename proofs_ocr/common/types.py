"""Shared data structures for proof OCR pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

Box = Tuple[int, int, int, int]


@dataclass
class SubproofNode:
    """Tree node describing one detected subproof scope region."""

    id: str
    box: Box
    children: List["SubproofNode"]
    parent: Optional["SubproofNode"] = None

    def __init__(self, box_id: str, box: Box) -> None:
        self.id = box_id
        self.box = box
        self.children = []
        self.parent = None

    @property
    def depth(self) -> int:
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth
