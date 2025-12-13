"""
Python port of the OpenLogicProject/fitch-checker syntax.php logic parser.
Original project licensed under GNU GPL v3.0. This module provides equivalent
functionality using Python classes and typing, preserving the recursive parsing
structure of the original PHP implementation.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import List, Optional


predicate_settings: bool = False


def has_stray_chars(s: str) -> bool:
    if predicate_settings:
        return re.match(r".*[^A-Za-z∀∃=¬∨∧↔→⊥\s\)\(\]\[\}\{]", s) is not None
    return re.match(r".*[^A-Z¬∨∧↔→⊥\s\)\(\]\[\}\{]", s) is not None


def regularize_me(s: str) -> str:
    s = re.sub(r"\[", "(", s)
    s = re.sub(r"\{", "(", s)
    s = re.sub(r"\]", ")", s)
    s = re.sub(r"\}", ")", s)
    s = re.sub(r"\s", "", s)
    return s


def list_union(z: List[str], x: List[str]) -> List[str]:
    return list(dict.fromkeys([*z, *x]))


@dataclass
class Wff:
    is_well_formed: bool = True
    err_msg: str = "none"
    wff_type: str = "unknown"
    main_op: str = "?"
    my_letter: str = ""
    left_side: Optional["Wff"] = None
    right_side: Optional["Wff"] = None
    my_terms: List[str] = field(default_factory=list)
    all_free_vars: List[str] = field(default_factory=list)


@dataclass
class WffStringDepth:
    string: str
    depth: int


def wff_to_string_and_depth(w: Wff) -> WffStringDepth:
    rv = WffStringDepth(string="", depth=0)
    if w.wff_type == "splat":
        rv.string = "⊥"
        return rv
    if w.wff_type == "identity":
        rv.string = f"{w.my_terms[0]} = {w.my_terms[1]}"
        return rv
    if w.wff_type == "atomic":
        rv.string = w.my_letter
        if predicate_settings:
            rv.string += "".join(w.my_terms)
        return rv
    if w.right_side is None:
        return rv
    right_result = wff_to_string_and_depth(w.right_side)
    right_add = right_result.string
    right_depth = right_result.depth
    if is_bin_op(w.right_side.main_op):
        if right_depth % 2 == 0:
            right_add = f"({right_add})"
        else:
            right_add = f"[{right_add}]"
        right_depth += 1
    if w.wff_type == "quantified":
        rv.string = f"{w.main_op}{w.my_letter}{right_add}"
        rv.depth = right_depth
        return rv
    if w.main_op == "¬":
        rv.string = f"¬{right_add}"
        rv.depth = right_depth
        return rv
    if w.left_side is None:
        return rv
    left_result = wff_to_string_and_depth(w.left_side)
    left_add = left_result.string
    left_depth = left_result.depth
    if is_bin_op(w.left_side.main_op):
        if left_depth % 2 == 0:
            left_add = f"({left_add})"
        else:
            left_add = f"[{left_add}]"
        left_depth += 1
    rv.depth = max(right_depth, left_depth)
    rv.string = f"{left_add} {w.main_op} {right_add}"
    return rv


def wff_to_string(w: Wff) -> str:
    return wff_to_string_and_depth(w).string


def is_bin_op(ch: str) -> bool:
    return ch in {"→", "∨", "∧", "↔"}


def is_mon_op(ch: str) -> bool:
    return ch in {"¬", "∀", "∃"}


def is_op(ch: str) -> bool:
    return is_bin_op(ch) or is_mon_op(ch)


def is_quantifier(ch: str) -> bool:
    return ch in {"∀", "∃"}


def is_var(ch: str) -> bool:
    return ch in {"x", "y", "z"}


def get_st_ltrs(w: Wff) -> List[str]:
    m_l: List[str] = []
    if w.wff_type == "splat":
        return m_l
    if w.wff_type == "atomic":
        return [w.my_letter]
    if w.main_op == "¬" and w.right_side is not None:
        return get_st_ltrs(w.right_side)
    if w.left_side is None or w.right_side is None:
        return m_l
    m_l = get_st_ltrs(w.left_side)
    x = get_st_ltrs(w.right_side)
    return list_union(m_l, x)


def new_sl_wff() -> Wff:
    w = Wff()
    w.left_side = Wff()
    w.right_side = Wff()
    return w


def new_pl_wff() -> Wff:
    w = new_sl_wff()
    w.my_terms = []
    w.all_free_vars = []
    return w


def parse_formula(s: str) -> Wff:
    global predicate_settings
    wff = new_pl_wff() if predicate_settings else new_sl_wff()
    main_op_pos = 0
    depth_array: List[int] = []
    s = regularize_me(s)

    if s == "":
        wff.is_well_formed = False
        wff.err_msg = "Formula or subformula is blank."
        return wff

    if has_stray_chars(s):
        wff.is_well_formed = False
        if predicate_settings:
            wff.err_msg = (
                "Input field contains characters or punctuation not allowed in the "
                "language of FOL. A statement should contain only parentheses ( [ { } ] ), "
                "predicates A–Z and =, terms a–w, variables x–z, the contradiction symbol ⊥, "
                "and the operators ¬, ∨, ∧, →, ↔, ∃, ∀ (or their alternatives)."
            )
        else:
            wff.err_msg = (
                "Input field contains characters or punctuation not allowed in the "
                "language of TFL. A statement should contain only parentheses ( [ { } ] ), "
                "statement letters A–Z, the contradiction symbol ⊥, and the operators ¬, ∨, ∧, →, "
                "and ↔ (or their alternatives)."
            )
        return wff

    d = 0
    for c in s:
        if c == "(":
            d += 1
        if c == ")":
            d -= 1
        depth_array.append(d)
    if depth_array[-1] != 0:
        wff.is_well_formed = False
        wff.err_msg = "Parentheses are unbalanced."
        return wff

    if depth_array[0] == 1:
        they_match = True
        for depth in depth_array[1:-1]:
            they_match = they_match and depth > 0
        if they_match:
            return parse_formula(s[1:-1])

    if not re.match(r".*[¬∧∨→↔∀∃]", s):
        if re.match(r".*[\(\)]", s):
            wff.is_well_formed = False
            wff.err_msg = "Misplaced parentheses."
            return wff
        if s == "⊥":
            wff.wff_type = "splat"
            return wff
        if re.match(r".*⊥", s):
            wff.is_well_formed = False
            wff.err_msg = "Formula contains ⊥ but not in isolation."
            return wff
        if re.match(r".*=", s):
            if re.match(r"[a-z]=[a-z]$", s):
                wff.wff_type = "identity"
                wff.my_terms = s.split("=")
                wff.all_free_vars = []
                if is_var(wff.my_terms[0]):
                    wff.all_free_vars.append(wff.my_terms[0])
                if is_var(wff.my_terms[1]) and wff.my_terms[1] != wff.my_terms[0]:
                    wff.all_free_vars.append(wff.my_terms[1])
                return wff
            wff.is_well_formed = False
            wff.err_msg = (
                "Poorly formed identity statement. Identity statement should be of the "
                "form <var>t</var> = <var>s</var>."
            )
            return wff
        if not predicate_settings:
            if re.match(r"[A-Z]$", s):
                wff.wff_type = "atomic"
                wff.my_letter = s
                return wff
            wff.is_well_formed = False
            wff.err_msg = (
                "Poorly formed atomic statement. In TFL, an atomic statement should be "
                "a single statement letter."
            )
            return wff
        if not re.match(r"[A-Z]", s):
            wff.is_well_formed = False
            wff.err_msg = "An atomic formula must begin with a predicate."
            return wff
        if len(s) == 1:
            wff.is_well_formed = False
            wff.err_msg = "An atomic formula must have terms, not just a predicate."
            return wff
        if re.match(r"..*[A-Z]", s):
            wff.is_well_formed = False
            wff.err_msg = "Predicates may only appear at the beginning of an atomic formula."
            return wff
        if re.match(r"..*[^a-z]", s):
            wff.is_well_formed = False
            wff.err_msg = "An atomic formula should contain only predicates followed by terms."
            return wff
        wff.wff_type = "atomic"
        wff.my_letter = s[0]
        wff.my_terms = list(s[1:])
        wff.all_free_vars = []
        for t in wff.my_terms:
            if is_var(t):
                wff.all_free_vars.append(t)
        wff.all_free_vars = list(dict.fromkeys(wff.all_free_vars))
        return wff

    for i, c in enumerate(s):
        if is_op(c) and depth_array[i] == 0:
            if wff.main_op == "?":
                wff.main_op = c
                main_op_pos = i
            else:
                if is_bin_op(wff.main_op) and is_bin_op(c):
                    wff.is_well_formed = False
                    wff.err_msg = "Too many operators or too few parentheses to disambiguate."
                    return wff
                if is_mon_op(wff.main_op) and is_bin_op(c):
                    wff.main_op = c
                    main_op_pos = i

    if wff.main_op == "?":
        wff.is_well_formed = False
        wff.err_msg = "Missing connective/operator or misplaced parentheses."
        return wff

    if is_quantifier(wff.main_op):
        wff.wff_type = "quantified"
        if main_op_pos != 0:
            wff.is_well_formed = False
            wff.err_msg = "Misuse of a quantifier internally in a formula."
            return wff
        if not is_var(s[1:2]):
            wff.is_well_formed = False
            wff.err_msg = "A quantifier is used without binding a variable."
        wff.my_letter = s[1:2]
        wff.right_side = parse_formula(s[2:])
        if not wff.right_side.is_well_formed:
            wff.is_well_formed = False
            wff.err_msg = wff.right_side.err_msg
            return wff
        wff.my_terms = wff.right_side.my_terms.copy()
        if wff.my_letter not in wff.my_terms:
            wff.my_terms.append(wff.my_letter)
        wff.all_free_vars = [v for v in wff.right_side.all_free_vars if v != wff.my_letter]
        return wff

    wff.wff_type = "molecular"

    if wff.main_op == "¬":
        if main_op_pos != 0:
            wff.is_well_formed = False
            wff.err_msg = "Misuse of negation internally in formula."
            return wff
        wff.right_side = parse_formula(s[1:])
        if not wff.right_side.is_well_formed:
            wff.is_well_formed = False
            wff.err_msg = wff.right_side.err_msg
            return wff
        if predicate_settings:
            wff.my_terms = wff.right_side.my_terms
            wff.all_free_vars = wff.right_side.all_free_vars
        return wff

    wff.left_side = parse_formula(s[:main_op_pos])
    if not wff.left_side.is_well_formed:
        wff.is_well_formed = False
        wff.err_msg = wff.left_side.err_msg
        return wff

    wff.right_side = parse_formula(s[main_op_pos + 1 :])
    if not wff.right_side.is_well_formed:
        wff.is_well_formed = False
        wff.err_msg = wff.right_side.err_msg
        return wff

    if predicate_settings:
        wff.my_terms = list_union(wff.left_side.my_terms, wff.right_side.my_terms)
        wff.all_free_vars = list_union(
            wff.left_side.all_free_vars, wff.right_side.all_free_vars
        )
    return wff


def sub_term(w: Wff, n: str, v: str) -> Wff:
    if v not in w.all_free_vars:
        return copy.deepcopy(w)
    x = new_pl_wff()
    x.wff_type = w.wff_type
    x.main_op = w.main_op
    x.my_letter = w.my_letter
    x.all_free_vars = [nv for nv in w.all_free_vars if nv != v]

    if w.wff_type in {"atomic", "identity"}:
        x.my_terms = [n if t == v else t for t in w.my_terms]
        return x

    if w.right_side is not None:
        x.right_side = sub_term(w.right_side, n, v)
        x.my_terms = x.right_side.my_terms
    if is_mon_op(x.main_op):
        return x
    if w.left_side is not None:
        x.left_side = sub_term(w.left_side, n, v)
        x.my_terms = list_union(x.my_terms, x.left_side.my_terms)
    return x


def same_wff(a: Wff, b: Wff) -> bool:
    if a.wff_type != b.wff_type:
        return False
    if a.wff_type == "splat":
        return True
    if a.wff_type == "identity":
        return (a.my_terms[0] == b.my_terms[0]) and (a.my_terms[1] == b.my_terms[1])
    if a.wff_type == "atomic":
        if predicate_settings:
            if a.my_letter != b.my_letter:
                return False
            if len(a.my_terms) != len(b.my_terms):
                return False
            for i, term in enumerate(a.my_terms):
                if term != b.my_terms[i]:
                    return False
            return True
        return a.my_letter == b.my_letter
    if a.main_op != b.main_op:
        return False
    if is_quantifier(a.main_op) and a.my_letter != b.my_letter:
        return False
    if is_mon_op(a.main_op):
        if a.right_side is None or b.right_side is None:
            return False
        return same_wff(a.right_side, b.right_side)
    if a.left_side is None or b.left_side is None or a.right_side is None or b.right_side is None:
        return False
    return same_wff(a.left_side, b.left_side) and same_wff(a.right_side, b.right_side)


__all__ = [
    "Wff",
    "WffStringDepth",
    "predicate_settings",
    "has_stray_chars",
    "regularize_me",
    "list_union",
    "wff_to_string_and_depth",
    "wff_to_string",
    "is_bin_op",
    "is_mon_op",
    "is_op",
    "is_quantifier",
    "is_var",
    "get_st_ltrs",
    "new_sl_wff",
    "new_pl_wff",
    "parse_formula",
    "sub_term",
    "same_wff",
]
