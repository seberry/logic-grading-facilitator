from __future__ import annotations

import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.fol_model_checker import check_sentences


DEFAULT_MODEL = """DOMAIN: {1,2,3}

CONSTANTS:
a: 1
b: 2

PREDICATES:
P: {1,2}
Q: {2}
R: {(1,2),(2,3)}
"""

DEFAULT_SENTENCES = """1. Pa
2. ∃x Rxa
3. ∀x (Px → Qx)
"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Print a sample FOL trace for a model/sentence pair.")
    parser.add_argument("--model-file", help="Path to a text file containing the model description.")
    parser.add_argument("--sentences-file", help="Path to a text file containing target sentences.")
    args = parser.parse_args()

    if args.model_file:
        with open(args.model_file, encoding="utf-8") as f:
            model_text = f.read()
    else:
        model_text = DEFAULT_MODEL

    if args.sentences_file:
        with open(args.sentences_file, encoding="utf-8") as f:
            sentences_text = f.read()
    else:
        sentences_text = DEFAULT_SENTENCES

    result = check_sentences(model_text, sentences_text)

    print("=== MODEL SUMMARY ===")
    print(result.model_summary)
    print()

    for item in result.sentence_results:
        print(f"=== SENTENCE {item.index}: {item.sentence} ===")
        print(item.trace)
        print()


if __name__ == "__main__":
    main()
