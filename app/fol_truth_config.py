"""Validate rubric-supplied finite models and normalize common FOL notation.

Normalization supplies one formula to both existing checkers; it does not
evaluate formulas or generate explanation wording.
"""
from __future__ import annotations

import json
import re


def object_token(value: int | str) -> str:
    # Keep numeric Quiz 9 readings compatible. Prefix named objects so objects
    # named "blank" or "na" cannot collide with reading-status values.
    return str(value) if isinstance(value, int) else "object:" + json.dumps(value, ensure_ascii=False)


def validate_model(model: object) -> dict:
    if not isinstance(model, dict) or set(model) - {"domain", "constants", "predicates"}:
        raise ValueError("FOL model must contain domain, constants, and predicates only.")
    domain = model.get("domain")
    if (not isinstance(domain, list) or not domain or
            any(isinstance(x, bool) or not isinstance(x, (int, str)) or
                (isinstance(x, int) and abs(x) > 2**53 - 1) or x == "" for x in domain)):
        raise ValueError("FOL domain must be a nonempty list of integer or nonempty string objects.")
    if len(set(domain)) != len(domain):
        raise ValueError("FOL domain contains duplicate objects.")
    constants, predicates = model.get("constants", {}), model.get("predicates", {})
    if not isinstance(constants, dict) or not isinstance(predicates, dict):
        raise ValueError("FOL constants and predicates must be mappings.")

    def in_domain(value):
        return not isinstance(value, bool) and isinstance(value, (int, str)) and value in domain

    for name, value in constants.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-t]", name):
            raise ValueError("FOL constants must be single letters a–t.")
        if not in_domain(value):
            raise ValueError(f"Constant {name} denotes an object outside the domain.")
    for name, extension in predicates.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Z]", name) or not isinstance(extension, list):
            raise ValueError("FOL predicates must be single uppercase letters with extension lists.")
        tuples = bool(extension) and isinstance(extension[0], list)
        arity = len(extension[0]) if tuples else 1
        if not arity:
            raise ValueError(f"Predicate {name} has an empty tuple.")
        for entry in extension:
            if isinstance(entry, list) != tuples:
                raise ValueError(f"Predicate {name} mixes objects and tuples.")
            members = entry if tuples else [entry]
            if len(members) != arity or not all(in_domain(x) for x in members):
                raise ValueError(f"Predicate {name} has inconsistent arity or an object outside the domain.")
    return {"domain": domain, "constants": constants, "predicates": predicates}


def checker_model(model: dict) -> dict:
    """Encode named objects for the existing integer-domain Python checker."""
    indices = {value: i + 1 for i, value in enumerate(model["domain"])}
    return {"domain": list(indices.values()),
            "constants": {name: indices[value] for name, value in model["constants"].items()},
            "predicates": {name: [[indices[x] for x in entry] if isinstance(entry, list) else indices[entry]
                                  for entry in extension]
                           for name, extension in model["predicates"].items()}}


def normalize_formula(source: object, domain_size: int = 1) -> str:
    """Match the browser's precedence and emit fully grouped compact notation.

    Both P(x)/R(x,y) and Px/Rxy are accepted, with x/y/z as variables.
    Quantifiers bind the next unary formula, as in the browser parser.
    """
    if not isinstance(source, str) or not source.strip() or len(source) > 2000:
        raise ValueError("Provide one FOL sentence of at most 2000 characters.")
    pattern = re.compile(r"\s*(<->|->|!=|forall\b|exists\b|[¬~!∧&∨|→↔∀∃≠=(),]|[A-Z]|[a-z]+)")
    tokens, position = [], 0
    source = source.strip()
    while position < len(source):
        match = pattern.match(source, position)
        if not match:
            raise ValueError(f"Unsupported FOL notation at character {position + 1}.")
        tokens.append(match[1])
        position = match.end()
    at = 0
    work = {}

    def record(formula, cost):
        if cost > 50_000:
            raise ValueError("FOL sentence exceeds the finite-model evaluation limit (50,000 trace nodes).")
        work[formula] = cost
        return formula

    def peek():
        return tokens[at] if at < len(tokens) else None

    def take():
        nonlocal at
        token = peek()
        at += 1
        return token

    def expect(token):
        if take() != token:
            raise ValueError(f"Expected {token!r} in FOL sentence.")

    def term(token):
        if token is None or not re.fullmatch(r"[a-txyz]", token):
            raise ValueError("Use single-letter constants a–t and variables x, y, z.")
        return token

    def unary():
        token = take()
        if token in ("¬", "~", "!"):
            body = unary()
            return record("¬" + body, 1 + work[body])
        if token in ("∀", "∃", "forall", "exists"):
            variable = take()
            if variable not in ("x", "y", "z"):
                raise ValueError("Quantified variables must be x, y, or z.")
            body = unary()
            return record(("∀" if token in ("∀", "forall") else "∃") + variable + body,
                          1 + domain_size * work[body])
        if token == "(":
            body = expression(0)
            expect(")")
            return body
        if token and re.fullmatch(r"[A-Z]", token):
            if peek() == "(":
                take()
                terms = [term(take())]
                while peek() == ",":
                    take()
                    terms.append(term(take()))
                expect(")")
            else:
                compact = take()
                if not compact or not re.fullmatch(r"[a-txyz]+", compact):
                    raise ValueError("A predicate needs terms, for example Px or R(x,y).")
                terms = list(compact)
            return record(token + "".join(terms), 1)
        left = term(token)
        operator = take()
        if operator not in ("=", "≠", "!="):
            raise ValueError("Expected an atomic FOL sentence or identity.")
        body = left + "=" + term(take())
        return record(body if operator == "=" else "¬" + body, 1 if operator == "=" else 2)

    operators = [(('↔', '<->'), '↔'), (('→', '->'), '→'), (('∨', '|'), '∨'), (('∧', '&'), '∧')]

    def expression(level):
        if level == len(operators):
            return unary()
        spellings, symbol = operators[level]
        left = expression(level + 1)
        while peek() in spellings:
            take()
            right = expression(level if symbol == '→' else level + 1)
            left = record(f"({left} {symbol} {right})", 1 + work[left] + work[right])
            if symbol == '↔' and peek() in spellings:
                raise ValueError("Parenthesize repeated biconditionals.")
        return left

    try:
        result = expression(0)
    except RecursionError:
        raise ValueError("FOL sentence is nested too deeply.") from None
    if at != len(tokens):
        raise ValueError(f"Unexpected token {peek()!r} in FOL sentence.")
    return result
