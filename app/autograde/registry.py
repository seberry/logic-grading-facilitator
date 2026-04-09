from __future__ import annotations

from typing import Callable, Dict

from . import fitch_proof, fol_model, kripke_model, modal_proof
from .base import AutoGradeResult

Engine = Callable[[str, dict], AutoGradeResult]


ENGINE_REGISTRY: Dict[str, Engine] = {
    "fitch_proof": fitch_proof.autograde,
    "fol_model": fol_model.autograde,
    "kripke_model": kripke_model.autograde,
    "modal_proof": modal_proof.autograde,
}


def get_engine(problem_type: str) -> Engine | None:
    return ENGINE_REGISTRY.get(problem_type)
