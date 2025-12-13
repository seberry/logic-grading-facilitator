"""
Python port of the OpenLogicProject/fitch-checker proofs.php verifier logic.
Original project licensed under GNU GPL v3.0; this translation preserves the
rule-checking semantics while adapting data structures to Python.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from app.syntax import (
    Wff,
    is_mon_op,
    is_var,
    same_wff,
    sub_term,
    predicate_settings,
    parse_formula,
)

TFL_RULES = [
    "∧I",
    "∧E",
    "⊥I",
    "⊥E",
    "¬I",
    "¬E",
    "→I",
    "→E",
    "TND",
    "∨I",
    "∨E",
    "↔I",
    "↔E",
    "DS",
    "R",
    "MT",
    "DNE",
    "DeM",
    "Pr",
    "Hyp",
    "X",
    "IP",
    "LEM",
]

FOL_RULES = ["∀E", "∀I", "∃I", "∃E", "=I", "=E", "CQ"]

CITE_NUMS = {
    "∧I": (2, 0),
    "∧E": (1, 0),
    "⊥I": (2, 0),
    "⊥E": (1, 0),
    "¬I": (0, 1),
    "¬E": (2, 0),
    "→I": (0, 1),
    "→E": (2, 0),
    "TND": (0, 2),
    "∨I": (1, 0),
    "∨E": (1, 2),
    "↔I": (0, 2),
    "↔E": (2, 0),
    "DS": (2, 0),
    "R": (1, 0),
    "MT": (2, 0),
    "DNE": (1, 0),
    "DeM": (1, 0),
    "∀E": (1, 0),
    "∀I": (1, 0),
    "∃I": (1, 0),
    "∃E": (1, 1),
    "=I": (0, 0),
    "=E": (2, 0),
    "CQ": (1, 0),
    "Hyp": (0, 0),
    "Pr": (0, 0),
    "X": (1, 0),
    "IP": (0, 1),
    "LEM": (0, 2),
}


class SubproofCitation(BaseModel):
    spstart: int
    spend: int


class Justification(BaseModel):
    rules: List[str] = Field(default_factory=list)
    lines: List[int] = Field(default_factory=list)
    subps: List[SubproofCitation] = Field(default_factory=list)
    parsed_ok: bool = True
    err_msg: str = ""


class ProofLine(BaseModel):
    wffstr: str
    jstr: str
    location: List[int]
    issues: List[str] = Field(default_factory=list)
    wff: Optional[Wff] = None
    j: Optional[Justification] = None
    can_be_checked: bool = True


@dataclass
class VerificationResult:
    issues: List[str]
    conc_reached: bool


def follows_by_cq_this_way(a: Wff, b: Wff) -> bool:
    return (
        (a.main_op == "¬")
        and (b.right_side is not None and b.right_side.main_op == "¬")
        and (
            (
                (a.right_side is not None)
                and (a.right_side.main_op == "∀")
                and (b.main_op == "∃")
            )
            or (
                (a.right_side is not None)
                and (a.right_side.main_op == "∃")
                and (b.main_op == "∀")
            )
        )
        and (b.my_letter == (a.right_side.my_letter if a.right_side else ""))
        and (
            a.right_side is not None
            and b.right_side is not None
            and same_wff(a.right_side.right_side, b.right_side.right_side)
        )
    )


def follows_by_cq(a: Wff, b: Wff) -> bool:
    return follows_by_cq_this_way(a, b) or follows_by_cq_this_way(b, a)


def is_self_id(w: Wff) -> bool:
    return (
        (w.wff_type == "identity") and (not is_var(w.my_terms[0])) and (w.my_terms[0] == w.my_terms[1])
    )


def differs_by_swapping_for(q: Wff, p: Wff, s: str, t: str) -> bool:
    if p.wff_type != q.wff_type:
        return False
    if p.wff_type == "splat":
        return True
    if p.wff_type in {"atomic", "identity"}:
        if len(p.my_terms) != len(q.my_terms):
            return False
        if p.wff_type == "atomic" and p.my_letter != q.my_letter:
            return False
        for idx, term in enumerate(p.my_terms):
            if term != q.my_terms[idx] and not (term == t and q.my_terms[idx] == s):
                return False
        return True
    if p.main_op != q.main_op:
        return False
    if is_mon_op(p.main_op):
        if p.right_side is None or q.right_side is None:
            return False
        return differs_by_swapping_for(q.right_side, p.right_side, s, t)
    return (
        p.right_side is not None
        and q.right_side is not None
        and p.left_side is not None
        and q.left_side is not None
        and differs_by_swapping_for(q.right_side, p.right_side, s, t)
        and differs_by_swapping_for(q.left_side, p.left_side, s, t)
    )


def follows_by_ll_this_way(c: Wff, a: Wff, b: Wff) -> bool:
    return (
        a.wff_type == "identity"
        and (
            differs_by_swapping_for(c, b, a.my_terms[0], a.my_terms[1])
            or differs_by_swapping_for(c, b, a.my_terms[1], a.my_terms[0])
        )
    )


def follows_by_ll(c: Wff, a: Wff, b: Wff) -> bool:
    return follows_by_ll_this_way(c, a, b) or follows_by_ll_this_way(c, b, a)


def follows_by_eg(c: Wff, a: Wff) -> bool:
    if c.main_op != "∃" or c.right_side is None:
        return False
    if c.my_letter not in c.right_side.all_free_vars:
        return same_wff(c.right_side, a)
    if c.my_letter in a.my_terms:
        return False
    for t in a.my_terms:
        if not is_var(t):
            if same_wff(a, sub_term(c.right_side, t, c.my_letter)):
                return True
    return False


def follows_by_ui(c: Wff, a: Wff) -> bool:
    if a.main_op != "∀" or a.right_side is None:
        return False
    for t in c.my_terms:
        if not is_var(t):
            if same_wff(c, sub_term(a.right_side, t, a.my_letter)):
                return True
    if a.my_letter not in a.right_side.all_free_vars and same_wff(c, a.right_side):
        return True
    return False


def follows_by_dem_this_way(a: Wff, b: Wff) -> bool:
    return (
        b.main_op == "¬"
        and b.right_side is not None
        and a.right_side is not None
        and a.left_side is not None
        and b.right_side.main_op in {"∨", "∧"}
        and ((a.main_op == "∧" and b.right_side.main_op == "∨") or (a.main_op == "∨" and b.right_side.main_op == "∧"))
        and a.right_side.main_op == "¬"
        and a.left_side.main_op == "¬"
        and b.right_side.right_side is not None
        and b.right_side.left_side is not None
        and same_wff(a.right_side.right_side, b.right_side.right_side)
        and same_wff(a.left_side.right_side, b.right_side.left_side)
    )


def follows_by_dem(c: Wff, a: Wff) -> bool:
    return follows_by_dem_this_way(c, a) or follows_by_dem_this_way(a, c)


def follows_by_dne(c: Wff, a: Wff) -> bool:
    return a.main_op == "¬" and a.right_side is not None and a.right_side.main_op == "¬" and same_wff(c, a.right_side.right_side)


def follows_by_mt_this_way(c: Wff, a: Wff, b: Wff) -> bool:
    return (
        a.main_op == "→"
        and b.main_op == "¬"
        and c.main_op == "¬"
        and a.right_side is not None
        and b.right_side is not None
        and same_wff(a.right_side, b.right_side)
        and a.left_side is not None
        and c.right_side is not None
        and same_wff(a.left_side, c.right_side)
    )


def follows_by_mt(c: Wff, a: Wff, b: Wff) -> bool:
    return follows_by_mt_this_way(c, a, b) or follows_by_mt_this_way(c, b, a)


def follows_by_ds_this_way(c: Wff, a: Wff, b: Wff) -> bool:
    return (
        a.main_op == "∨"
        and b.main_op == "¬"
        and a.left_side is not None
        and a.right_side is not None
        and b.right_side is not None
        and (
            (same_wff(b.right_side, a.right_side) and same_wff(c, a.left_side))
            or (same_wff(b.right_side, a.left_side) and same_wff(c, a.right_side))
        )
    )


def follows_by_ds(c: Wff, a: Wff, b: Wff) -> bool:
    return follows_by_ds_this_way(c, a, b) or follows_by_ds_this_way(c, b, a)


def follows_by_conj_intro_this_way(rw: Wff, a: Wff, b: Wff) -> bool:
    return rw.main_op == "∧" and rw.right_side is not None and rw.left_side is not None and same_wff(rw.right_side, a) and same_wff(rw.left_side, b)


def follows_by_conj_intro(rw: Wff, a: Wff, b: Wff) -> bool:
    return follows_by_conj_intro_this_way(rw, a, b) or follows_by_conj_intro_this_way(rw, b, a)


def follows_by_conj_elim(rw: Wff, a: Wff) -> bool:
    return a.main_op == "∧" and a.right_side is not None and a.left_side is not None and (same_wff(a.right_side, rw) or same_wff(a.left_side, rw))


def follows_by_contra_intro(c: Wff, a: Wff, b: Wff) -> bool:
    return c.wff_type == "splat" and (
        (b.main_op == "¬" and b.right_side is not None and same_wff(a, b.right_side))
        or (a.main_op == "¬" and a.right_side is not None and same_wff(b, a.right_side))
    )


def follows_by_mp_this_way(c: Wff, a: Wff, b: Wff) -> bool:
    return (
        a.main_op == "→"
        and a.right_side is not None
        and same_wff(a.right_side, c)
        and a.left_side is not None
        and same_wff(a.left_side, b)
    )


def follows_by_mp(c: Wff, a: Wff, b: Wff) -> bool:
    return follows_by_mp_this_way(c, a, b) or follows_by_mp_this_way(c, b, a)


def follows_by_cp(c: Wff, a: Wff, b: Wff) -> bool:
    return c.main_op == "→" and c.left_side is not None and c.right_side is not None and same_wff(c.left_side, a) and same_wff(c.right_side, b)


def follows_by_raa(c: Wff, a: Wff, b: Wff) -> bool:
    return c.main_op == "¬" and c.right_side is not None and same_wff(c.right_side, a) and b.wff_type == "splat"


def follows_by_ip(c: Wff, a: Wff, b: Wff) -> bool:
    return a.main_op == "¬" and a.right_side is not None and same_wff(a.right_side, c) and b.wff_type == "splat"


def follows_by_tnd_this_way(c: Wff, i: Wff, j: Wff, k: Wff, l: Wff) -> bool:
    return (
        k.main_op == "¬"
        and k.right_side is not None
        and same_wff(k.right_side, i)
        and same_wff(j, l)
        and same_wff(c, j)
    )


def follows_by_tnd(c: Wff, i: Wff, j: Wff, k: Wff, l: Wff) -> bool:
    return follows_by_tnd_this_way(c, i, j, k, l) or follows_by_tnd_this_way(c, k, l, i, j)


def follows_by_add(c: Wff, a: Wff) -> bool:
    return c.main_op == "∨" and c.left_side is not None and c.right_side is not None and (same_wff(c.left_side, a) or same_wff(c.right_side, a))


def follows_by_disj_elim_this_way(c: Wff, m: Wff, i: Wff, j: Wff, k: Wff, l: Wff) -> bool:
    return (
        m.main_op == "∨"
        and m.left_side is not None
        and m.right_side is not None
        and same_wff(m.left_side, i)
        and same_wff(m.right_side, k)
        and same_wff(j, l)
        and same_wff(j, c)
    )


def follows_by_disj_elim(c: Wff, m: Wff, i: Wff, j: Wff, k: Wff, l: Wff) -> bool:
    return follows_by_disj_elim_this_way(c, m, i, j, k, l) or follows_by_disj_elim_this_way(c, m, k, l, i, j)


def follows_by_bicon_intro_this_way(c: Wff, i: Wff, j: Wff, k: Wff, l: Wff) -> bool:
    return (
        c.main_op == "↔"
        and c.left_side is not None
        and c.right_side is not None
        and same_wff(c.left_side, i)
        and same_wff(c.right_side, j)
        and same_wff(c.right_side, k)
        and same_wff(c.left_side, l)
    )


def follows_by_bicon_intro(c: Wff, i: Wff, j: Wff, k: Wff, l: Wff) -> bool:
    return follows_by_bicon_intro_this_way(c, i, j, k, l) or follows_by_bicon_intro_this_way(c, k, l, i, j)


def follows_by_bicon_elim_this_way(c: Wff, a: Wff, b: Wff) -> bool:
    return (
        a.main_op == "↔"
        and a.left_side is not None
        and a.right_side is not None
        and ((same_wff(a.left_side, b) and same_wff(a.right_side, c)) or (same_wff(a.left_side, c) and same_wff(a.right_side, b)))
    )


def follows_by_bicon_elim(c: Wff, a: Wff, b: Wff) -> bool:
    return follows_by_bicon_elim_this_way(c, a, b) or follows_by_bicon_elim_this_way(c, b, a)


def new_j() -> Justification:
    return Justification()


def parse_j(jstr: str) -> Justification:
    j = new_j()
    cleaned = re.sub(r"[;,\s]+", ",", jstr)
    cleaned = re.sub(r"[-–]+", "-", cleaned)
    parts = cleaned.split(",")
    for part in parts:
        if part == "":
            j.parsed_ok = False
            j.err_msg = "Justification left blank."
            return j
        if re.fullmatch(r"[0-9]*", part):
            j.lines.append(int(part))
            continue
        if re.fullmatch(r"[0-9]+-[0-9]+", part):
            start, end = part.split("-")
            spc = SubproofCitation(spstart=int(start), spend=int(end))
            j.subps.append(spc)
            continue
        if part in TFL_RULES or (predicate_settings and part in FOL_RULES):
            j.rules.append(part)
        else:
            j.parsed_ok = False
            j.err_msg = f"Justification cites nonexistent rule ({part}) or is badly formed."
            return j
    if len(j.rules) > 1:
        j.parsed_ok = False
        j.err_msg = "More than one rule cited."
    if len(j.rules) < 1:
        j.parsed_ok = False
        j.err_msg = "No rule cited."
    return j


def flatten_proof(pr: List[Union[ProofLine, List]], dpth_ar: List[int]) -> List[ProofLine]:
    fpr: List[ProofLine] = []
    for i, element in enumerate(pr):
        if isinstance(element, list):
            fpr.extend(flatten_proof(element, [*dpth_ar, i]))
        else:
            x = element.copy(deep=True)
            x.location = [*dpth_ar, i]
            x.issues = []
            fpr.append(x)
    return fpr


def check_proof(pr: List[Union[ProofLine, List]], numprems: int, conc: str) -> VerificationResult:
    rv = VerificationResult(issues=[], conc_reached=False)
    fpr = flatten_proof(pr, [])

    for line in fpr:
        line.wff = parse_formula(line.wffstr)
        if not line.wff.is_well_formed:
            line.issues.append(f"Not well-formed: {line.wff.err_msg}")

    for line in fpr:
        line.j = parse_j(line.jstr)
        if not line.j.parsed_ok:
            line.issues.append(f"Cannot parse justification: {line.j.err_msg}")

    for line in fpr:
        if line.j and line.j.parsed_ok:
            good_lc, good_spc = CITE_NUMS[line.j.rules[0]]
            act_lc = len(line.j.lines)
            act_spc = len(line.j.subps)
            if act_lc < good_lc:
                line.issues.append(f"Cites too few line numbers for the rule {line.j.rules[0]}.")
            if act_lc > good_lc:
                line.issues.append(f"Cites too many line numbers for the rule {line.j.rules[0]}.")
            if act_spc < good_spc:
                line.issues.append(f"Cites too few ranges of lines for the rule {line.j.rules[0]}.")
            if act_spc > good_spc:
                line.issues.append(f"Cites too many ranges of lines for the rule {line.j.rules[0]}.")

    for i, line in enumerate(fpr):
        if line.j and line.j.parsed_ok:
            n = i + 1
            nloc = line.location
            for citedline in line.j.lines:
                if citedline > len(fpr) or citedline < 1:
                    line.issues.append(f"Cites nonexistent line ({citedline}).")
                    continue
                if citedline == n:
                    line.issues.append("Cites itself.")
                    continue
                if citedline > n:
                    line.issues.append(f"Cites a line ({citedline}) that occurs after it.")
                    continue
                cloc = fpr[citedline - 1].location
                if len(cloc) > len(nloc):
                    line.issues.append(f"Cites an unavailable line ({citedline}).")
                    continue
                problem = False
                for d in range(len(cloc) - 1):
                    if cloc[d] != nloc[d]:
                        problem = True
                        break
                if problem:
                    line.issues.append(f"Cites an unavailable line ({citedline}).")
                    continue
            for citedsp in line.j.subps:
                startcite = citedsp.spstart
                endcite = citedsp.spend
                if startcite > endcite:
                    line.issues.append(f"Cites a range of lines in the wrong order ({startcite}–{endcite}).")
                    continue
                if startcite > len(fpr) or endcite > len(fpr) or startcite < 1 or endcite < 0:
                    line.issues.append(f"Cites a line nonexistent range of lines ({startcite}–{endcite}).")
                    continue
                if endcite >= n:
                    line.issues.append(f"Cites a line range after or including itself ({startcite}–{endcite}).")
                    continue
                endloc = fpr[endcite - 1].location
                startloc = fpr[startcite - 1].location
                problem = len(endloc) != len(startloc) or startloc[-1] != 0
                for l_idx in range(len(startloc) - 1):
                    if endloc[l_idx] != startloc[l_idx]:
                        problem = True
                        break
                if problem:
                    line.issues.append(f"Cites a range of lines which do not make up a subproof ({startcite}–{endcite}).")
                    continue
                cloc = startloc[:-1]
                if len(cloc) > len(nloc) or len(cloc) < len(nloc):
                    line.issues.append(f"Cites an unavailable subproof ({startcite}–{endcite}).")
                    continue
                problem = False
                for d in range(len(cloc) - 1):
                    if cloc[d] != nloc[d]:
                        problem = True
                        break
                if problem:
                    line.issues.append(f"Cites an unavailable subproof ({startcite}–{endcite}).")

    for i, line in enumerate(fpr):
        line.can_be_checked = True
        if line.issues:
            line.can_be_checked = False
            continue
        if line.j is None:
            line.can_be_checked = False
            continue
        for cl in line.j.lines:
            cn = cl - 1
            if not fpr[cn].wff or not fpr[cn].wff.is_well_formed:
                line.can_be_checked = False
                line.issues.append(f"Cites another line that is not well-formed ({cl}).")
        for csp in line.j.subps:
            csn = csp.spstart - 1
            cen = csp.spend - 1
            if not fpr[csn].wff or not fpr[csn].wff.is_well_formed:
                line.can_be_checked = False
                line.issues.append(f"Cites another line that is not well-formed ({csp.spstart}).")
            if not fpr[cen].wff or not fpr[cen].wff.is_well_formed:
                line.can_be_checked = False
                line.issues.append(f"Cites another line that is not well-formed ({csp.spend}).")

    for i, line in enumerate(fpr):
        if not line.can_be_checked or not line.j:
            continue
        worked = False
        rule = line.j.rules[0]
        if rule == "Pr":
            worked = (i + 1) <= numprems
        elif rule == "Hyp":
            worked = line.location[-1] == 0
        elif rule == "∧I":
            worked = follows_by_conj_intro(line.wff, fpr[line.j.lines[0] - 1].wff, fpr[line.j.lines[1] - 1].wff)
        elif rule == "∧E":
            worked = follows_by_conj_elim(line.wff, fpr[line.j.lines[0] - 1].wff)
        elif rule == "⊥E":
            worked = fpr[line.j.lines[0] - 1].wff.wff_type == "splat"
        elif rule == "⊥I":
            worked = follows_by_contra_intro(line.wff, fpr[line.j.lines[0] - 1].wff, fpr[line.j.lines[1] - 1].wff)
        elif rule == "→E":
            worked = follows_by_mp(line.wff, fpr[line.j.lines[0] - 1].wff, fpr[line.j.lines[1] - 1].wff)
        elif rule == "→I":
            worked = follows_by_cp(line.wff, fpr[line.j.subps[0].spstart - 1].wff, fpr[line.j.subps[0].spend - 1].wff)
        elif rule == "¬I":
            worked = follows_by_raa(line.wff, fpr[line.j.subps[0].spstart - 1].wff, fpr[line.j.subps[0].spend - 1].wff)
        elif rule == "¬E":
            worked = follows_by_contra_intro(line.wff, fpr[line.j.lines[0] - 1].wff, fpr[line.j.lines[1] - 1].wff)
        elif rule == "IP":
            worked = follows_by_ip(line.wff, fpr[line.j.subps[0].spstart - 1].wff, fpr[line.j.subps[0].spend - 1].wff)
        elif rule in {"TND", "LEM"}:
            worked = follows_by_tnd(
                line.wff,
                fpr[line.j.subps[0].spstart - 1].wff,
                fpr[line.j.subps[0].spend - 1].wff,
                fpr[line.j.subps[1].spstart - 1].wff,
                fpr[line.j.subps[1].spend - 1].wff,
            )
        elif rule == "∨I":
            worked = follows_by_add(line.wff, fpr[line.j.lines[0] - 1].wff)
        elif rule == "∨E":
            worked = follows_by_disj_elim(
                line.wff,
                fpr[line.j.lines[0] - 1].wff,
                fpr[line.j.subps[0].spstart - 1].wff,
                fpr[line.j.subps[0].spend - 1].wff,
                fpr[line.j.subps[1].spstart - 1].wff,
                fpr[line.j.subps[1].spend - 1].wff,
            )
        elif rule == "↔I":
            worked = follows_by_bicon_intro(
                line.wff,
                fpr[line.j.subps[0].spstart - 1].wff,
                fpr[line.j.subps[0].spend - 1].wff,
                fpr[line.j.subps[1].spstart - 1].wff,
                fpr[line.j.subps[1].spend - 1].wff,
            )
        elif rule == "↔E":
            worked = follows_by_bicon_elim(line.wff, fpr[line.j.lines[0] - 1].wff, fpr[line.j.lines[1] - 1].wff)
        elif rule == "X":
            worked = fpr[line.j.lines[0] - 1].wff.wff_type == "splat"
        elif rule == "DS":
            worked = follows_by_ds(line.wff, fpr[line.j.lines[0] - 1].wff, fpr[line.j.lines[1] - 1].wff)
        elif rule == "R":
            worked = same_wff(line.wff, fpr[line.j.lines[0] - 1].wff)
        elif rule == "MT":
            worked = follows_by_mt(line.wff, fpr[line.j.lines[0] - 1].wff, fpr[line.j.lines[1] - 1].wff)
        elif rule == "DNE":
            worked = follows_by_dne(line.wff, fpr[line.j.lines[0] - 1].wff)
        elif rule == "DeM":
            worked = follows_by_dem(line.wff, fpr[line.j.lines[0] - 1].wff)
        elif rule == "∀E":
            worked = follows_by_ui(line.wff, fpr[line.j.lines[0] - 1].wff)
        elif rule == "∃I":
            worked = follows_by_eg(line.wff, fpr[line.j.lines[0] - 1].wff)
        elif rule == "∀I":
            worked = False
            univ = line.wff
            if univ.main_op == "∀" and univ.right_side is not None:
                inst = fpr[line.j.lines[0] - 1].wff
                bound_var = univ.my_letter
                if bound_var in univ.right_side.all_free_vars:
                    for t in inst.my_terms:
                        if t in univ.my_terms:
                            continue
                        if not is_var(t) and same_wff(inst, sub_term(univ.right_side, t, bound_var)):
                            found = False
                            for j in range(i):
                                if fpr[j].j and fpr[j].j.rules[0] in {"Pr", "Hyp"}:
                                    hyp_loc = fpr[j].location
                                    this_loc = line.location
                                    if len(hyp_loc) > len(this_loc):
                                        continue
                                    problem = False
                                    for d in range(len(hyp_loc) - 1):
                                        if hyp_loc[d] != this_loc[d]:
                                            problem = True
                                            break
                                    if not problem and t in fpr[j].wff.my_terms:
                                        found = True
                                        break
                            if found:
                                continue
                            worked = True
                else:
                    worked = same_wff(univ.right_side, inst)
        elif rule == "∃E":
            worked = False
            exwff = fpr[line.j.lines[0] - 1].wff
            if exwff.main_op == "∃" and exwff.right_side is not None:
                sp_hyp = fpr[line.j.subps[0].spstart - 1].wff
                sp_res = fpr[line.j.subps[0].spend - 1].wff
                res = line.wff
                if same_wff(sp_res, res):
                    if exwff.my_letter in exwff.right_side.all_free_vars:
                        for t in sp_hyp.my_terms:
                            if is_var(t):
                                continue
                            if same_wff(sp_hyp, sub_term(exwff.right_side, t, exwff.my_letter)):
                                if t in res.my_terms or t in exwff.my_terms:
                                    continue
                                found = False
                                for j in range(i):
                                    if fpr[j].j and fpr[j].j.rules[0] in {"Pr", "Hyp"}:
                                        hyp_loc = fpr[j].location
                                        this_loc = line.location
                                        if len(hyp_loc) > len(this_loc):
                                            continue
                                        problem = False
                                        for d in range(len(hyp_loc) - 1):
                                            if hyp_loc[d] != this_loc[d]:
                                                problem = True
                                                break
                                        if not problem and t in fpr[j].wff.my_terms:
                                            found = True
                                            break
                                if found:
                                    continue
                                worked = True
                    else:
                        worked = same_wff(exwff.right_side, sp_hyp)
        elif rule == "=I":
            worked = is_self_id(line.wff)
        elif rule == "=E":
            worked = follows_by_ll(line.wff, fpr[line.j.lines[0] - 1].wff, fpr[line.j.lines[1] - 1].wff)
        elif rule == "CQ":
            worked = follows_by_cq(line.wff, fpr[line.j.lines[0] - 1].wff)
        if not worked:
            line.issues.append(f"Is not a proper application of the rule {rule} (for the line(s) cited).")

    for idx, l in enumerate(fpr):
        n = idx + 1
        for issue in l.issues:
            rv.issues.append(f"Line {n}: {issue}")

    if not rv.issues:
        conc_wff = parse_formula(conc)
        if not conc_wff.is_well_formed:
            rv.issues.append("Desired conclusion is not a wff. Oops!")
        else:
            for line in fpr:
                if len(line.location) == 1 and same_wff(line.wff, conc_wff):
                    rv.conc_reached = True
    return rv


__all__ = [
    "ProofLine",
    "Justification",
    "SubproofCitation",
    "VerificationResult",
    "flatten_proof",
    "check_proof",
]
